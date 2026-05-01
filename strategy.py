#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Uses 1w EMA50 to filter regime: price > EMA50 = bullish bias (long bias), price < EMA50 = bearish bias (short bias)
# Donchian channel (20-period high/low) from 1d OHLC acts as breakout levels
# Breakout above upper band with volume spike = long, breakdown below lower band with volume spike = short
# Volume spike defined as current volume > 1.5 * 20-period EMA
# Designed for low frequency (30-100 trades over 4 years) with clear structure
# Works in both bull and bear markets via trend filter that adapts bias

name = "1d_Donchian20_1wEMA50_Volume_Bias_v1"
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
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band = 20-period low
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d levels to 1d timeframe (no shift needed as we use completed candles)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # 1d volume spike filter: volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(50, 20)  # Need EMA50 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA50 = bullish bias, price < EMA50 = bearish bias
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above upper band with volume spike AND bullish bias
            if close[i] > upper_20_aligned[i] and volume_spike[i] and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band with volume spike AND bearish bias
            elif close[i] < lower_20_aligned[i] and volume_spike[i] and bearish_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to midpoint of Donchian channel or opposite breakout
            midpoint = (upper_20_aligned[i] + lower_20_aligned[i]) / 2.0
            exit_long = False
            if close[i] <= midpoint:  # Return to midpoint
                exit_long = True
            elif close[i] < lower_20_aligned[i] and volume_spike[i]:  # Reverse breakout
                exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to midpoint of Donchian channel or opposite breakout
            midpoint = (upper_20_aligned[i] + lower_20_aligned[i]) / 2.0
            exit_short = False
            if close[i] >= midpoint:  # Return to midpoint
                exit_short = True
            elif close[i] > upper_20_aligned[i] and volume_spike[i]:  # Reverse breakout
                exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals