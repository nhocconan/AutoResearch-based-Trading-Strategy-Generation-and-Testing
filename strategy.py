#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 12h Camarilla R3 AND price > 12h EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 AND price < 12h EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 12h Camarilla H3/L3 levels OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Camarilla levels from 12h provide intraday support/resistance; 12h EMA50 filters trend; volume confirms breakout
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for Camarilla levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (using previous 12h bar's OHLC)
    # Camarilla: based on previous period's range
    prev_close = df_12h['close'].shift(1).values
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    r3 = prev_close + (prev_range * 1.1 / 4)
    s3 = prev_close - (prev_range * 1.1 / 4)
    h3 = prev_close + (prev_range * 1.1 / 6)
    l3 = prev_close - (prev_range * 1.1 / 6)
    r4 = prev_close + (prev_range * 1.1 / 2)
    s4 = prev_close - (prev_range * 1.1 / 2)
    
    # Align 12h Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3, above 12h EMA50, volume confirmation
            if close[i] > r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, below 12h EMA50, volume confirmation
            elif close[i] < s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below H3 OR volume drops below average
            if close[i] < h3_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above L3 OR volume drops below average
            if close[i] > l3_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals