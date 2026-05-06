#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakouts with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 on 4h AND 1d EMA50 > EMA50 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 1h
# Short when price breaks below Camarilla S3 on 4h AND 1d EMA50 < EMA50 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 1h
# Exit when price retouches the Camarilla pivot point (mean reversion to center)
# Uses discrete sizing 0.20 to minimize fee churn and manage drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide precise intraday support/resistance in ranging markets
# 1d EMA50 trend filter ensures we trade with the dominant daily trend
# Volume spike confirmation (2.0x) validates breakout strength while limiting overtrading
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least 2 completed 4h bars for Camarilla (requires high/low/close)
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for 4h timeframe
    # Camarilla pivot point = (high + low + close) / 3
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    # Camarilla ranges
    range_4h = high_4h - low_4h
    # Resistance levels
    r1_4h = pp_4h + (range_4h * 1.0/12)
    r2_4h = pp_4h + (range_4h * 2.0/12)
    r3_4h = pp_4h + (range_4h * 3.0/12)
    r4_4h = pp_4h + (range_4h * 4.0/12)
    # Support levels
    s1_4h = pp_4h - (range_4h * 1.0/12)
    s2_4h = pp_4h - (range_4h * 2.0/12)
    s3_4h = pp_4h - (range_4h * 3.0/12)
    s4_4h = pp_4h - (range_4h * 4.0/12)
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 on 4h, 1d EMA50 > EMA50 previous (uptrend), volume spike, in session
            if (close[i] > r3_4h_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 on 4h, 1d EMA50 < EMA50 previous (downtrend), volume spike, in session
            elif (close[i] < s3_4h_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retouches the Camarilla pivot point (mean reversion)
            if close[i] <= pp_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retouches the Camarilla pivot point (mean reversion)
            if close[i] >= pp_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals