#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_1d_camarilla_breakout_v1
# Uses Camarilla pivot levels from 12h timeframe with 1d trend filter for breakouts.
# Breakouts occur when price breaks above R4 or below S4 with volume confirmation.
# In ranging markets, fades at R3/S3 levels. Designed to work in both bull and bear
# markets by adapting to volatility regimes. Low frequency expected due to strict
# breakout/fade criteria and multi-timeframe confirmation.
name = "6h_12h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_12h = (high_12h + low_12h + close_12h) / 3
    pivot_12h = typical_12h
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = pivot_12h + (range_12h * 1.1 / 2)
    r4_12h = pivot_12h + (range_12h * 1.1)
    s3_12h = pivot_12h - (range_12h * 1.1 / 2)
    s4_12h = pivot_12h - (range_12h * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any data not ready
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        bullish_breakout = (
            close[i] > r4_12h_aligned[i] and  # price breaks above R4
            vol_confirm  # volume confirmation
        )
        
        bearish_breakout = (
            close[i] < s4_12h_aligned[i] and  # price breaks below S4
            vol_confirm  # volume confirmation
        )
        
        # Fade conditions (mean reversion at R3/S3)
        bullish_fade = (
            close[i] <= s3_12h_aligned[i] and  # price at or below S3
            close[i] > ema_50_aligned[i] and   # but above 1d EMA50 (bullish bias)
            vol_confirm
        )
        
        bearish_fade = (
            close[i] >= r3_12h_aligned[i] and  # price at or above R3
            close[i] < ema_50_aligned[i] and   # but below 1d EMA50 (bearish bias)
            vol_confirm
        )
        
        # Exit conditions: opposite signal or volatility collapse
        vol_collapse = volume[i] < 0.5 * vol_ma_20[i]
        
        exit_long = bearish_breakout or bearish_fade or vol_collapse
        exit_short = bullish_breakout or bullish_fade or vol_collapse
        
        # Determine signal direction based on 1d trend
        bullish_bias = close[i] > ema_50_aligned[i]
        bearish_bias = close[i] < ema_50_aligned[i]
        
        # Long signals: breakout in uptrend OR fade at support in any trend
        long_signal = (bullish_breakout and bullish_bias) or bullish_fade
        # Short signals: breakout in downtrend OR fade at resistance in any trend
        short_signal = (bearish_breakout and bearish_bias) or bearish_fade
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals