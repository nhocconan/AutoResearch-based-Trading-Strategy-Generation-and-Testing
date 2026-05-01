#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Uses 1w EMA50 to filter trend: price > EMA50 = bullish bias (long breakouts), price < EMA50 = bearish bias (short breakouts)
# Donchian channel (20-period high/low) from 1d acts as structure for breakouts
# Breakout above upper band with volume spike = long, breakdown below lower band with volume spike = short
# Volume spike defined as current volume > 2.0 * 20-period average (high threshold to reduce trades)
# Designed for very low frequency (<50 trades over 4 years) to minimize fee drag and maximize edge

name = "1d_Donchian20_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w HTF data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band = 20-period high
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band = 20-period low
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d levels to 1d timeframe (no change needed but for consistency)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # 1d volume spike filter: volume > 2.0 * 20-period EMA (high threshold)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)  # Need EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        bullish_bias = close_1w.iloc[i] > ema_50_aligned[i] if hasattr(close_1w, 'iloc') else close_1w[i] > ema_50_aligned[i]
        bearish_bias = close_1w.iloc[i] < ema_50_aligned[i] if hasattr(close_1w, 'iloc') else close_1w[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above upper band with volume spike in bullish bias
            if bullish_bias and close[i] > upper_band_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with volume spike in bearish bias
            elif bearish_bias and close[i] < lower_band_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (upper_band_aligned[i] + lower_band_aligned[i]) / 2.0
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (upper_band_aligned[i] + lower_band_aligned[i]) / 2.0
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals