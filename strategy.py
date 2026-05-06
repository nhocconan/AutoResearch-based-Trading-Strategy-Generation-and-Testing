#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla R3/S3 levels with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 12h Camarilla R3 level AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 6h
# Short when price breaks below 12h Camarilla S3 level AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 6h
# Exit when price returns to 12h Camarilla pivot level (mean reversion to center)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Camarilla R3/S3 provides high-probability breakout/continuation points
# 1d EMA34 trend filter ensures we trade with the dominant daily trend
# Volume spike confirmation (2.0x) validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 2 completed 12h bars for pivot calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (using previous 12h bar)
    # Pivot = (High + Low + Close) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Range = High - Low
    range_12h = high_12h - low_12h
    # Camarilla levels
    r3_12h = pivot_12h + (range_12h * 1.1 / 4.0)  # R3 = pivot + 1.1*range/4
    s3_12h = pivot_12h - (range_12h * 1.1 / 4.0)  # S3 = pivot - 1.1*range/4
    # Additional levels for exit
    pp_12h = pivot_12h  # Pivot point for exit
    
    # Align 12h Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    
    # Calculate 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or np.isnan(pp_12h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h R3, 1d EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (close[i] > r3_12h_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S3, 1d EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (close[i] < s3_12h_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h pivot (mean reversion)
            if close[i] <= pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h pivot (mean reversion)
            if close[i] >= pp_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals