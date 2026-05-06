#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels (R3/S3) breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 12h Camarilla R3 AND 12h EMA50 > EMA50 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below 12h Camarilla S3 AND 12h EMA50 < EMA50 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price crosses back through 12h Camarilla pivot point (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide precise reversal zones in ranging markets
# 12h EMA50 trend filter ensures we trade with the dominant intermediate trend
# Volume spike confirmation (2.0x) validates breakout strength while limiting overtrading
# Works in both bull (buy breakouts above R3) and bear (sell breakdowns below S3) markets

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
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
    if len(df_12h) < 2:  # Need at least 2 completed 12h bars for Camarilla
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels: PP = (H+L+C)/3, Range = H-L
    # R3 = PP + Range * 1.1/2, S3 = PP - Range * 1.1/2
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    pivot_point_12h = typical_price_12h
    camarilla_r3_12h = pivot_point_12h + (range_12h * 1.1 / 2.0)
    camarilla_s3_12h = pivot_point_12h - (range_12h * 1.1 / 2.0)
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    pivot_point_aligned = align_htf_to_ltf(prices, df_12h, pivot_point_12h)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(pivot_point_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, 12h EMA50 > EMA50 previous (uptrend), volume spike, in session
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, 12h EMA50 < EMA50 previous (downtrend), volume spike, in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below pivot point (mean reversion)
            if close[i] < pivot_point_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above pivot point (mean reversion)
            if close[i] > pivot_point_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals