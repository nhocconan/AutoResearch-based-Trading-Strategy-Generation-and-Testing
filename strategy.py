#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Long when price breaks above 1h Camarilla R3 AND price > 4h EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1h Camarilla S3 AND price < 4h EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 1h Camarilla pivot point OR volume < avg_volume(20)
# Uses discrete sizing 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels from 1h provide intraday support/resistance; 4h EMA50 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla levels (based on previous bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 for entries, pivot point (PP) for exits
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # handle first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    rang = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pp + 1.1 * rang
    camarilla_s3 = camarilla_pp - 1.1 * rang
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 4h EMA50, volume confirmation
            if close[i] > camarilla_r3[i] and close[i] > ema50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3, below 4h EMA50, volume confirmation
            elif close[i] < camarilla_s3[i] and close[i] < ema50_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla pivot point OR volume drops below average
            if close[i] < camarilla_pp[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price crosses above Camarilla pivot point OR volume drops below average
            if close[i] > camarilla_pp[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals