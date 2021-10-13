from setuptools import setup, find_packages

REQUIRED = ['gym', 'numpy', 'pandas', 'ray[rllib]']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='sumo-rl',
    version='2.3',
    packages=['sumo_rl',],
    install_requires=REQUIRED,
    author='LucasAlegre',
    author_email='lucasnale@gmail.com',
    url='https://github.com/KMASAHIRO/sumo-rl',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    description='Environments inheriting OpenAI Gym Env and RL algorithms for Traffic Signal Control on SUMO.'
)
