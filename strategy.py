#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and volume spike (>1.8x average)
# Uses 4h for signal direction (EMA50 trend) and 1d for volume regime filter (avoid low volume periods)
# Entry only during 08-20 UTC session to reduce noise. Discrete sizing 0.20 to minimize fee churn.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_Donchian20_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume average for regime filter (avoid low volume periods)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: 1.8x 20-period average (strict to avoid overtrading)
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian HIGH AND price > 4h EMA50 AND volume spike AND 1d volume above average
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i] and 
                volume[i] > vol_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Donchian LOW AND price < 4h EMA50 AND volume spike AND 1d volume above average
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i] and 
                  volume[i] > vol_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian LOW OR price < 4h EMA50
            if close[i] < lowest_low[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian HIGH OR price > 4h EMA50
            if close[i] > highest_high[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals