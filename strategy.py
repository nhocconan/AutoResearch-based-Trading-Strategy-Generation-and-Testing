#!/usr/bin/env python3
"""
exp_6551_6h_donchian20_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot levels as directional filter.
Long when price > weekly pivot and breaks above Donchian HIGH; short when price < weekly pivot and breaks below Donchian LOW.
Volume confirmation (1.5x 20-period MA) reduces false breakouts.
Weekly pivot provides structural support/resistance that works in both bull and bear markets.
Designed for 50-150 total trades over 4 years with discrete sizing (0.25) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6551_6h_donchian20_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.5      # volume must be 1.5x its 20-period MA
SIGNAL_SIZE = 0.25       # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (HLC of previous week)
    # We approximate weekly pivot using rolling window on daily data
    # Weekly high = max(high) over prior 5 trading days (approximation)
    # Weekly low = min(low) over prior 5 trading days
    # Weekly close = close of prior day (simplified)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-day roll (approximates 1 week)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align to LTF (6h) with shift(1) for completed bars only
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 5  # extra for weekly calc
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]):
            continue
            
        # Long conditions: price > weekly pivot + breaks above Donchian HIGH + volume confirmation
        long_bias = close[i] > weekly_pivot_aligned[i]  # price above weekly pivot (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < weekly pivot + breaks below Donchian LOW + volume confirmation
        short_bias = close[i] < weekly_pivot_aligned[i]  # price below weekly pivot (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: pivot reversal or Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below weekly pivot (trend change)
            exit_long = close[i] < weekly_pivot_aligned[i]
            # Or if price drops below Donchian midpoint
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above weekly pivot (trend change)
            exit_short = close[i] > weekly_pivot_aligned[i]
            # Or if price rises above Donchian midpoint
            exit_short = exit_short or close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>
#!/usr/bin/env python3
"""
exp_6551_6h_donchian20_1d_pivot_v1
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot levels as directional filter.
Long when price > weekly pivot and breaks above Donchian HIGH; short when price < weekly pivot and breaks below Donchian LOW.
Volume confirmation (1.5x 20-period MA) reduces false breakouts.
Weekly pivot provides structural support/resistance that works in both bull and bear markets.
Designed for 50-150 total trades over 4 years with discrete sizing (0.25) to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6551_6h_donchian20_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.5      # volume must be 1.5x its 20-period MA
SIGNAL_SIZE = 0.25       # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (HLC of previous week)
    # We approximate weekly pivot using rolling window on daily data
    # Weekly high = max(high) over prior 5 trading days (approximation)
    # Weekly low = min(low) over prior 5 trading days
    # Weekly close = close of prior day (simplified)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-day roll (approximates 1 week)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).apply(lambda x: x[-1]).values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align to LTF (6h) with shift(1) for completed bars only
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD) + 5  # extra for weekly calc
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]):
            continue
            
        # Long conditions: price > weekly pivot + breaks above Donchian HIGH + volume confirmation
        long_bias = close[i] > weekly_pivot_aligned[i]  # price above weekly pivot (bullish)
        long_breakout = close[i] > donchian_high[i-1]  # break above previous period's high
        long_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Short conditions: price < weekly pivot + breaks below Donchian LOW + volume confirmation
        short_bias = close[i] < weekly_pivot_aligned[i]  # price below weekly pivot (bearish)
        short_breakout = close[i] < donchian_low[i-1]  # break below previous period's low
        short_volume = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: pivot reversal or Donchian midpoint reversal
        if position == 1:  # long position
            # Exit if price drops back below weekly pivot (trend change)
            exit_long = close[i] < weekly_pivot_aligned[i]
            # Or if price drops below Donchian midpoint
            exit_long = exit_long or close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price rises back above weekly pivot (trend change)
            exit_short = close[i] > weekly_pivot_aligned[i]
            # Or if price rises above Donchian midpoint
            exit_short = exit_short or close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            if long_bias and long_breakout and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_bias and short_breakout and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals

</think>