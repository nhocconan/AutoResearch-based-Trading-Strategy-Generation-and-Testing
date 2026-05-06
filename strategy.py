#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above 1w Camarilla S4 level AND 1d EMA34 is rising AND 6h volume > 1.8 * avg_volume(24)
# Short when price breaks below 1w Camarilla R4 level AND 1d EMA34 is falling AND 6h volume > 1.8 * avg_volume(24)
# Exit when price returns to 1w Camarilla midpoint (PP) or opposite extreme (S3/R3)
# Uses discrete sizing 0.28 to balance profit potential and drawdown control
# Target: 60-120 total trades over 4 years (15-30/year) for 6h timeframe
# Weekly Camarilla provides strong structural levels from higher timeframe
# 1d EMA34 ensures we trade with the daily trend while reducing noise
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "6h_1wCamarilla_S4_R4_Breakout_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least 1 completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    # Camarilla levels
    r3_1w = pp_1w + range_1w * 1.1 / 4
    r4_1w = pp_1w + range_1w * 1.1 / 2
    s3_1w = pp_1w - range_1w * 1.1 / 4
    s4_1w = pp_1w - range_1w * 1.1 / 2
    
    # Align 1w Camarilla levels to 6h timeframe (wait for completed 1w bar)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 24-period average volume on 6h
    avg_volume_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(avg_volume_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla S4 level, EMA34 rising, volume spike
            if (close[i] > s4_1w_aligned[i] and close[i-1] <= s4_1w_aligned[i-1] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.28
                position = 1
            # Short: price breaks below 1w Camarilla R4 level, EMA34 falling, volume spike
            elif (close[i] < r4_1w_aligned[i] and close[i-1] >= r4_1w_aligned[i-1] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price returns to 1w Camarilla midpoint (PP) or below S3
            if close[i] <= pp_1w_aligned[i] or close[i] <= s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price returns to 1w Camarilla midpoint (PP) or above R3
            if close[i] >= pp_1w_aligned[i] or close[i] >= r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals