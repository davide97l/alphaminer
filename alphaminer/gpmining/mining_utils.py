import graphviz
import matplotlib.pyplot as plt

from qlib.data import D


def get_data(market, fields, start_time, end_time, return_shifts=[5]):
    instruments = D.instruments(market=market)

    for r in return_shifts:
        fields.append('Ref($close, -{})/$close - 1'.format(r))

    all_data = D.features(
        instruments=instruments, 
        fields=fields, 
        start_time=start_time,
        end_time=end_time
    )
    all_data = all_data.rename(
        columns={'Ref($close, -{})/$close - 1'.format(r): 'return_{}'.format(r) for r in return_shifts}
    )
    all_data = all_data.dropna(axis=0)
    # all_data = all_data.interpolate()

    return all_data


def draw_statistics(statistics, name):
    duration = statistics['generation_time'].sum() / 60
    x = statistics['generation']
    plt.figure(2, figsize=(10, 7))
    plt.plot(x, statistics['average_fitness'], label='average')
    plt.plot(x, statistics['best_fitness'], label='best')
    plt.plot(x, statistics['best_oob_fitness'], label='best_oob')
    plt.legend(loc='best')
    plt.savefig('gp_curve_{}.png'.format(name))
    # plt.show()


def draw_alpha_graph(alpha, name):
    print('fitness: {}; depth: {}; length: {}'.format(alpha.fitness_, alpha.depth_, alpha.length_))
    dot_data = alpha.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('alpha_graph_{}'.format(name), format='svg', cleanup=True)
    return graph
