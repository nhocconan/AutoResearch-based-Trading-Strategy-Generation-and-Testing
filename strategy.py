#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above 20-period high AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Short when price breaks below 20-period low AND 1d ADX > 25 AND volume > 1.5x 20-period average.
# Exit when price crosses below 10-period EMA (for long) or above 10-period EMA (for short).
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

name = "12h_Donchian20_1dADX25_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d ADX (14-period) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr3 = pd.Series(high_1d) - pd.Series(close_1d).shift(1)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).shift(1) - pd.Series(low_1d)
    up_move = up_move.where((up_move > down_move) & (up_move > 0), 0)
    down_move = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    plus_di = 100 * (up_move.ewm(alpha=1/14, adjust=False).mean() / atr)
    minus_di = 100 * (down_move.ewm(alpha=1/14, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 10-period EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(ema10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above 20-period high, ADX > 25, volume spike
            long_cond = (close[i] > highest_high[i]) and (adx_1d_aligned[i] > 25) and volume_filter[i]
            # Short conditions: breakout below 20-period low, ADX > 25, volume spike
            short_cond = (close[i] < lowest_low[i]) and (adx_1d_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 10-period EMA
            if close[i] < ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 10-period EMA
            if close[i] > ema10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals