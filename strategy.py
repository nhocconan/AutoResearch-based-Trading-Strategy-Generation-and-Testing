#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Donchian breakout with 1w trend filter and volume confirmation
# In bull/bear markets: Donchian(20) breakout on 1d captures major moves; 1w EMA50 filter ensures we trade with higher timeframe trend
# Volume confirmation reduces false breakouts. Discrete sizing 0.25 limits trades to ~12-37/year.
# Works in bull markets (breakouts up) and bear markets (breakouts down) with trend filter preventing counter-trend entries.

name = "6h_1d_1w_donchian_breakout_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period) - use prior close to avoid look-ahead
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d and 1w indicators to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian lower band or trend turns bearish
            if close[i] < lowest_20_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian upper band or trend turns bullish
            if close[i] > highest_20_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above upper band with volume confirmation and bullish trend
            if close[i] > highest_20_aligned[i] and volume_confirmed and close[i] > ema50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below lower band with volume confirmation and bearish trend
            elif close[i] < lowest_20_aligned[i] and volume_confirmed and close[i] < ema50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals