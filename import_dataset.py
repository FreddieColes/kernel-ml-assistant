import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Dict
import mysql.connector
from mysql.connector import Error
from rich.console import Console
from rich.progress import Progress, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

def fix_or_create_config():
    config_file = Path('config.json')    
    if config_file.exists():
            with open(config_file, 'r') as f:
                content = f.read().strip()
                if content:
                    config = json.loads(content)
                    console.print("[green]Found existing valid configuration[/green]")
                    return config.get('mysql_config', None)
                else:
                    console.print("[yellow]config.json is empty[/yellow]")      
    #Test connection
    try:
        conn = mysql.connector.connect(
            host=mysql_config['host'],
            user=mysql_config['user'],
            password=mysql_config['password']
        )
        conn.close()
        console.print("[green]Successfully connected to MySQL[/green]")
        
        #Save configuration
        full_config = {
            'mysql_config': mysql_config,
        }
        
        with open(config_file, 'w') as f:
            json.dump(full_config, f, indent=2)
        return mysql_config

    except Error as e:
        console.print(f"[red]Failed to connect: {e}[/red]")
        return None




class KernelDataImporter:
    #Import the kernel patch dataset into MySQL
    
    def __init__(self, mysql_config):
        self.console = console
        self.mysql_config = mysql_config
        
    def create_database(self):
        try:
            conn = mysql.connector.connect(
                host=self.mysql_config['host'],
                user=self.mysql_config['user'],
                password=self.mysql_config['password']
            )
            cursor = conn.cursor()            
            cursor.execute("CREATE DATABASE IF NOT EXISTS kernel_patch2")
            cursor.execute("USE kernel_patch2")
            
            #Set max packets for large imports
            cursor.execute("SET GLOBAL max_allowed_packet=1073741824")            
            self.console.print("[green]Database 'kernel_patch2' ready[/green]")            
            cursor.close()
            conn.close()
            return True
        except Error as e:
            self.console.print(f"[red]Error creating database: {e}[/red]")
            return False
    
    def import_sql_files(self):
        #Import all SQL dump files
        sql_dir = Path('./datasets/level-1 and level-2')
        
        if not sql_dir.exists():
            self.console.print(f"[red]✗ Directory not found: {sql_dir}[/red]")
            self.console.print("\nPlease ensure you have extracted the dataset files to:")
            self.console.print("  ./datasets/level-1 and level-2/")
            self.console.print("  ./datasets/level-0/raw/")
            return False
        
        #Define import order (tables with dependencies last)
        import_order = [
            'kernel_patch2_patch.sql',
            'kernel_patch2_commit.sql',
            'kernel_patch2_patch_comment.sql',
            'kernel_patch2_commit_to_patch.sql',
            'kernel_patch2_patch_set.sql',
            'kernel_patch2_patch_set_patchwork_ids.sql',
            'kernel_patch2_my_author.sql',
            'kernel_patch2_my_author_author_email.sql',
            'kernel_patch2_my_author_author_name.sql',
            'kernel_patch2_patch_author.sql'
        ]
        
        #Find all SQL files
        sql_files = list(sql_dir.glob('kernel_patch2_*.sql'))        
        if not sql_files:
            self.console.print(f"[red]✗ No SQL files found in {sql_dir}[/red]")
            return False
        sorted_files = []
        for filename in import_order:
            filepath = sql_dir / filename
            if filepath.exists():
                sorted_files.append(filepath)
        for filepath in sql_files:
            if filepath not in sorted_files:
                sorted_files.append(filepath)        
        self.console.print(f"\n[blue]Found {len(sorted_files)} SQL files to import[/blue]")
        
        #Check if mysql command exists
        try:
            subprocess.run(['mysql', '--version'], capture_output=True, check=True)
        except:
            #If mysql command doesn't work, try alternative import method
            self.console.print("[yellow]Using Python import method[/yellow]")
            return self.import_sql_files_python(sorted_files)
        
        #Import each file using mysql command
        success_count = 0
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("[blue]Importing SQL files...", total=len(sorted_files))
            
            for sql_file in sorted_files:
                progress.update(task, description=f"[blue]Importing {sql_file.name}...")
                
                cmd = [
                    'mysql',
                    '-h', self.mysql_config['host'],
                    '-u', self.mysql_config['user'],
                    f'-p{self.mysql_config["password"]}',
                    'kernel_patch2',
                    '-e', f'source {sql_file.as_posix()}'
                ]
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        if "already exists" in result.stderr:
                            self.console.print(f"[red]Table already exists in {sql_file.name}[/red]")
                        else:
                            self.console.print(f"[yellow]Warning importing {sql_file.name}: {result.stderr[:100]}[/yellow]")
                    else:
                        self.console.print(f"[green]Imported {sql_file.name}[/green]")
                        success_count += 1
                
                except Exception as e:
                    self.console.print(f"[red]Error importing {sql_file.name}: {e}[/red]")
                
                progress.update(task, advance=1)
        
        return success_count > 0
    
    def import_sql_files_python(self, sorted_files):
        #Import SQL files using Python (fallback method)
        self.console.print("\n[blue]Using Python import method...[/blue]")
        
        success_count = 0
        conn = mysql.connector.connect(**self.mysql_config)
        cursor = conn.cursor()
        
        for sql_file in sorted_files:
            self.console.print(f"Importing {sql_file.name}...")
            
            try:
                with open(sql_file, 'r', encoding='utf-8') as f:
                    sql_content = f.read()
                
                #Split by semicolons but be careful with strings
                statements = []
                current = []
                in_string = False
                escape_next = False
                
                for line in sql_content.split('\n'):
                    #Skip comments and empty lines
                    if line.strip().startswith('--') or line.strip().startswith('/*') or not line.strip():
                        continue
                    
                    current.append(line)
                    
                    # imple check for statement end
                    if ';' in line and not in_string:
                        statements.append('\n'.join(current))
                        current = []
                
                #Execute each statement
                for stmt in statements:
                    if stmt.strip():
                        try:
                            cursor.execute(stmt)
                        except Error as e:
                            if "already exists" not in str(e):
                                self.console.print(f"[yellow]Statement error: {str(e)[:100]}[/yellow]")
                
                conn.commit()
                self.console.print(f"[green]✓ Imported {sql_file.name}[/green]")
                success_count += 1
                
            except Exception as e:
                self.console.print(f"[red]✗ Error with {sql_file.name}: {e}[/red]")
        
        cursor.close()
        conn.close()
        
        return success_count > 0
    
    def verify_import(self):
        #Verify the import was successful
        self.console.print("\n[bold blue]Verifying import...[/bold blue]")
        
        try:
            conn = mysql.connector.connect(**self.mysql_config)
            cursor = conn.cursor()
            
            #Check main tables
            tables_to_check = [
                ('patch', 'Patches'),
                ('patch_comment', 'Comments'),
                ('commit_to_patch', 'Commit mappings'),
                ('my_author', 'Unique authors'),
                ('patch_set', 'Patch sets')
            ]
            
            total_records = 0
            for table, description in tables_to_check:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    total_records += count
                    self.console.print(f"  {description}: [bold]{count:,}[/bold] records")
                except Error:
                    self.console.print(f"  {description}: [yellow]Table not found[/yellow]")
            
            cursor.close()
            conn.close()
            
            if total_records > 0:
                self.console.print("\n[green]Import verification complete![/green]")
                return True
            else:
                self.console.print("\n[red]No data found in tables[/red]")
                return False
                
        except Exception as e:
            self.console.print(f"[red]Verification failed: {e}[/red]")
            return False

def main():  
    #Fix or create configuration
    mysql_config = fix_or_create_config()
    if not mysql_config:
        console.print("\n[red]Failed to configure MySQL connection[/red]")
        sys.exit(1)
    
    #Run import
    importer = KernelDataImporter(mysql_config)
    
    if not importer.create_database():
        sys.exit(1)
    
    if not importer.import_sql_files():
        console.print("\n[red]Import failed[/red]")
        sys.exit(1)
    
    if not importer.verify_import():
        console.print("\n[yellow]Import completed with warnings[/yellow]")
    
    console.print("\n[green]✅ Setup complete![/green]")
    console.print("\nYou can now run: [blue]python patch_analysis.py[/blue]")

if __name__ == "__main__":
    main()