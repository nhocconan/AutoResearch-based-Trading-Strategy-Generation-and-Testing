#!/usr/bin/env python3
"""
exp_7507_6h_donchian20_1w_pivot_volume_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- In bull regime (price > weekly pivot): buy Donchian(20) breakouts on volume surge
- In bear regime (price < weekly pivot): sell Donchian(20) breakdowns on volume surge
- Weekly pivot acts as trend filter to avoid counter-trend entries
- Volume surge confirms institutional participation
- Targets 75-150 trades over 4 years (19-37/year) with strict breakout conditions
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7507_6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SURGE_MULTIPLIER = 1.5  # 1.5x average volume
VOLUME_LOOKBACK = 20
PIVOT_LOOKBACK = 5  # for swing high/low
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: (H + L + C) / 3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime using weekly pivot
        bull_regime = close[i] > weekly_pivot_aligned[i]   # price above weekly pivot
        bear_regime = close[i] < weekly_pivot_aligned[i]   # price below weekly pivot
        
        # Volume confirmation
        volume_surge = volume[i] > (VOLUME_SURGE_MULTIPLIER * vol_avg[i])
        
        # Donchian breakout/breakdown signals
        donchian_breakout = high[i] > highest_high[i-1]  # break above previous period's high
        donchian_breakdown = low[i] < lowest_low[i-1]    # break below previous period's low
        
        # Entry conditions
        long_entry = (
            bull_regime and           # bull regime (above weekly pivot)
            donchian_breakout and     # Donchian breakout
            volume_surge              # volume confirmation
        )
        
        short_entry = (
            bear_regime and           # bear regime (below weekly pivot)
            donchian_breakdown and    # Donchian breakdown
            volume_surge              # volume confirmation
        )
        
        # Exit conditions - reverse signal or stoploss (handled above)
        long_exit = donchian_breakdown  # exit long on breakdown
        short_exit = donchian_breakout  # exit short on breakout
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals

</think>

#!/usr/bin/env python3
"""
exp_7507_6h_donchian20_1w_pivot_volume_v1
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- In bull regime (price > weekly pivot): buy Donchian(20) breakouts on volume surge
- In bear regime (price < weekly pivot): sell Donchian(20) breakdowns on volume surge
- Weekly pivot acts as trend filter to avoid counter-trend entries
- Volume surge confirms institutional participation
- Targets 75-150 trades over 4 years (19-37/year) with strict breakout conditions
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7507_6h_donchian20_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_SURGE_MULTIPLIER = 1.5  # 1.5x average volume
VOLUME_LOOKBACK = 20
PIVOT_LOOKBACK = 5  # for swing high/low
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: (H + L + C) / 3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    vol_avg = pd.Series(volume).rolling(window=VOLUME_LOOKBACK, min_periods=VOLUME_LOOKBACK).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_LOOKBACK, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(weekly_pivot_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market regime using weekly pivot
        bull_regime = close[i] > weekly_pivot_aligned[i]   # price above weekly pivot
        bear_regime = close[i] < weekly_pivot_aligned[i]   # price below weekly pivot
        
        # Volume confirmation
        volume_surge = volume[i] > (VOLUME_SURGE_MULTIPLIER * vol_avg[i])
        
        # Donchian breakout/breakdown signals
        donchian_breakout = high[i] > highest_high[i-1]  # break above previous period's high
        donchian_breakdown = low[i] < lowest_low[i-1]    # break below previous period's low
        
        # Entry conditions
        long_entry = (
            bull_regime and           # bull regime (above weekly pivot)
            donchian_breakout and     # Donchian breakout
            volume_surge              # volume confirmation
        )
        
        short_entry = (
            bear_regime and           # bear regime (below weekly pivot)
            donchian_breakdown and    # Donchian breakdown
            volume_surge              # volume confirmation
        )
        
        # Exit conditions - reverse signal or stoploss (handled above)
        long_exit = donchian_breakdown  # exit long on breakdown
        short_exit = donchian_breakout  # exit short on breakout
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals