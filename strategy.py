#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation
# Long when price breaks above 12h Camarilla R3 AND price > 12h EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 AND price < 12h EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 12h Camarilla H3/L3 levels OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Camarilla levels from 12h provide intraday support/resistance; 12h EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_VolumeSpike"
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
    
    # Get 12h data ONCE before loop for Camarilla levels and EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for calculations
        return np.zeros(n)
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Camarilla levels (based on previous 12h bar)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # H3 = close + 1.1*(high-low)*1.1/6, L3 = close - 1.1*(high-low)*1.1/6
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = close_12h[0]  # Avoid NaN on first bar
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    diff_12h = prev_high_12h - prev_low_12h
    camarilla_r3 = prev_close_12h + 1.1 * diff_12h * 1.1 / 4
    camarilla_s3 = prev_close_12h - 1.1 * diff_12h * 1.1 / 4
    camarilla_h3 = prev_close_12h + 1.1 * diff_12h * 1.1 / 6
    camarilla_l3 = prev_close_12h - 1.1 * diff_12h * 1.1 / 6
    
    # Align 12h Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 12h EMA34, volume confirmation
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 12h EMA34, volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla H3 OR volume drops below average
            if close[i] < camarilla_h3_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla L3 OR volume drops below average
            if close[i] > camarilla_l3_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals