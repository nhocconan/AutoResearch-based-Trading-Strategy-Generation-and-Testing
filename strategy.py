#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R3 level AND 12h EMA50 > EMA50 previous (uptrend) AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 12h Camarilla S3 level AND 12h EMA50 < EMA50 previous (downtrend) AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price retests the 12h Camarilla pivot point (PP)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 12h Camarilla provides strong intraday pivot levels with high reversal/continuation probability
# 12h EMA50 ensures we trade with the dominant 12h trend filter
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "4h_Camarilla_R3S3_12hEMA50_Trend_Volume"
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
    
    # Get 12h data ONCE before loop for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 2 completed 12h bars for pivot calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r3_12h = pp_12h + (high_12h - low_12h) * 1.1 / 2.0
    s3_12h = pp_12h - (high_12h - low_12h) * 1.1 / 2.0
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3, 12h EMA50 > EMA50 previous (uptrend), volume spike, in session
            if (close[i] > r3_aligned[i] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3, 12h EMA50 < EMA50 previous (downtrend), volume spike, in session
            elif (close[i] < s3_aligned[i] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 12h Camarilla pivot point
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 12h Camarilla pivot point
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals