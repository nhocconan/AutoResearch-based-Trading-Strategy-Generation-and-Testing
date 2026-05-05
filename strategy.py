#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Camarilla pivot levels (S3/R3) with 12h EMA50 trend filter and volume spike confirmation
# Long when price touches/bounces off S3 AND price > 12h EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price touches/rejects R3 AND price < 12h EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses the daily pivot point (PP) OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla pivots provide precise support/resistance levels that work in ranging markets
# 12h EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms reversal strength and reduces false signals
# Works in bull markets (buying S3 bounces in uptrend) and bear markets (selling R3 rejections in downtrend)

name = "12h_Camarilla_S3R3_12hEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Based on previous day's OHLC
    pp = (high_1d + low_1d + close_1d) / 3
    r1 = pp + (high_1d - low_1d) * 1.08333 / 12
    s1 = pp - (high_1d - low_1d) * 1.08333 / 12
    r2 = pp + (high_1d - low_ld) * 1.08333 / 6
    s2 = pp - (high_1d - low_1d) * 1.08333 / 6
    r3 = pp + (high_1d - low_1d) * 1.08333 / 4
    s3 = pp - (high_1d - low_1d) * 1.08333 / 4
    r4 = pp + (high_1d - low_1d) * 1.08333 / 2
    s4 = pp - (high_1d - low_1d) * 1.08333 / 2
    
    # Align Camarilla levels to 12h timeframe (wait for completed daily bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches/bounces off S3 AND above 12h EMA50 AND volume confirmation
            if (low[i] <= s3_aligned[i] * 1.001 and close[i] > s3_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price touches/rejects R3 AND below 12h EMA50 AND volume confirmation
            elif (high[i] >= r3_aligned[i] * 0.999 and close[i] < r3_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses above daily pivot OR volume drops below average
            if close[i] > pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses below daily pivot OR volume drops below average
            if close[i] < pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals