import click
from psychohistory.core.engine import run_simulation  # Example import

@click.group()
def cli():
    """Psychohistory CLI Tool"""
    pass

@cli.command()
@click.option('--data', required=True, help="Path to input data file.")
def analyze(data):
    """Run psychohistory analysis on the given data file."""
    click.echo(f"Running psychohistory analysis on: {data}")
    result = run_simulation(data)
    click.echo(f"Analysis complete: {result}")

if __name__ == "__main__":
    cli()
