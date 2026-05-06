#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R3 AND 1d EMA34 is rising AND 4h volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 AND 1d EMA34 is falling AND 4h volume > 1.5 * avg_volume(20)
# Exit when price returns to 12h Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# 12h Camarilla provides structure from higher timeframe, reducing noise
# 1d EMA34 ensures we trade with the daily trend while reducing whipsaws
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "4h_12hCamarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 2 completed 12h bars for Camarilla (requires high/low/close)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (R3, S3, PP)
    # Camarilla formulas:
    # PP = (high + low + close) / 3
    # R3 = PP + (high - low) * 1.1 / 4
    # S3 = PP - (high - low) * 1.1 / 4
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r3_12h = pp_12h + (high_12h - low_12h) * 1.1 / 4.0
    s3_12h = pp_12h - (high_12h - low_12h) * 1.1 / 4.0
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3, EMA34 rising, volume spike
            if (close[i] > r3_12h_aligned[i] and close[i-1] <= r3_12h_aligned[i-1] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3, EMA34 falling, volume spike
            elif (close[i] < s3_12h_aligned[i] and close[i-1] >= s3_12h_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h Camarilla pivot point (PP)
            if close[i] <= pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h Camarilla pivot point (PP)
            if close[i] >= pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals