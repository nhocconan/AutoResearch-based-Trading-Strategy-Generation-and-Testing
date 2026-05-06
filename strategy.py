#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla R3 AND 12h EMA34 > EMA34 previous (uptrend) AND volume > 1.8 * avg_volume(20) on 4h
# Short when price breaks below 4h Camarilla S3 AND 12h EMA34 < EMA34 previous (downtrend) AND volume > 1.8 * avg_volume(20) on 4h
# Exit when price crosses back through the 4h Camarilla midpoint (R3/S3 average)
# Uses discrete sizing 0.30 to balance return and risk
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# 4h Camarilla R3/S3 provides strong breakout levels that reduce whipsaw vs tighter levels
# 12h EMA34 trend filter ensures we trade with the dominant intermediate trend
# Volume confirmation (1.8x) validates breakout strength while limiting overtrading
# Works in bull (trend + breakouts) and bear (mean reversion at extremes via exits)

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
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
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least one completed 4h bar
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (R3, S3, midpoint)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_low_4h = high_4h - low_4h
    camarilla_r3_4h = close_4h + 1.1 * high_low_4h * 1.1 / 4.0
    camarilla_s3_4h = close_4h - 1.1 * high_low_4h * 1.1 / 4.0
    camarilla_mid_4h = (camarilla_r3_4h + camarilla_s3_4h) / 2.0
    
    # Align 4h Camarilla to 4h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid_4h)
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need at least 34 completed 12h bars for EMA34
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R3, 12h EMA34 > EMA34 previous (uptrend), volume confirmation, in session
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 4h Camarilla S3, 12h EMA34 < EMA34 previous (downtrend), volume confirmation, in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 4h Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back above 4h Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals