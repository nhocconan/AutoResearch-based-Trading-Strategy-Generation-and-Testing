#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla R3 AND 4h EMA50 > EMA100 AND volume > 1.8 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND 4h EMA50 < EMA100 AND volume > 1.8 * avg_volume(20)
# Exit when price crosses 4h EMA50 (trend reversal signal)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Camarilla provides strong daily structure with clear breakout/fade levels
# 4h EMA50/EMA100 filter ensures alignment with intermediate trend
# Volume confirmation filters weak breakouts
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend)

name = "4h_1dCamarillaR3S3_4hEMA50Trend_Volume_v1"
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
    
    # Get 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous 1d bar)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 0.275*(high-low), S3 = close - 0.275*(high-low)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3_1d = typical_price_1d + 0.275 * range_1d
    camarilla_s3_1d = typical_price_1d - 0.275 * range_1d
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 100:  # Need sufficient data for EMA100
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 and EMA100 for trend filter
    close_series_4h = pd.Series(close_4h)
    ema_50_4h = close_series_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_4h = close_series_4h.ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Align 4h EMA indicators to 4h timeframe (wait for completed 4h bar)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_100_aligned = align_htf_to_ltf(prices, df_4h, ema_100_4h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_100_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla R3 with 4h EMA50 > EMA100 and volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and close[i-1] <= camarilla_r3_aligned[i-1] and 
                ema_50_aligned[i] > ema_100_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla S3 with 4h EMA50 < EMA100 and volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and close[i-1] >= camarilla_s3_aligned[i-1] and 
                  ema_50_aligned[i] < ema_100_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 (trend reversal)
            if close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 (trend reversal)
            if close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals