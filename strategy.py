# 12h_1d_donchian_breakout_v1
# Strategy: 12h Donchian breakout with daily trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Breakouts of 20-period Donchian channels capture trends. Daily trend filter (EMA50) ensures alignment with higher timeframe momentum. Volume confirmation filters false breakouts. Works in bull by catching breakouts in uptrend and in bear by catching breakdowns in downtrend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = close_1d > ema_50_1d  # Uptrend when close > EMA50
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 20-period Donchian channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_break = close[i] > high_max[i-1] and trend_1d_aligned[i] and vol_spike[i]
        short_break = close[i] < low_min[i-1] and not trend_1d_aligned[i] and vol_spike[i]
        
        # Exit conditions: opposite breakout
        exit_long = position == 1 and close[i] < low_min[i-1]
        exit_short = position == -1 and close[i] > high_max[i-1]
        
        # Trading logic
        if long_break and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_break and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals