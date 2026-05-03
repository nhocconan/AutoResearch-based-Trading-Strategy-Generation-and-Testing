#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Camarilla levels provide intraday support/resistance structure. Breakout at R3/S3 with
# volume spike and alignment to 12h EMA50 trend captures momentum moves while avoiding
# false breakouts in ranging markets. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via upward breaks at R3 and in bear markets via downward breaks at S3.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 6h bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    typical_price = (high + low + close) / 3.0
    typical_price_series = pd.Series(typical_price)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    prev_high = high_series.shift(1)
    prev_low = low_series.shift(1)
    prev_close = typical_price_series.shift(1)  # Use typical price for pivot calculation
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    camarilla_r3 = pivot + 1.125 * range_hl
    camarilla_s3 = pivot - 1.125 * range_hl
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Camarilla and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend alignment with volume spike
            if close[i] > camarilla_r3[i] and ema_50_12h_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in downtrend alignment with volume spike
            elif close[i] < camarilla_s3[i] and ema_50_12h_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses uptrend alignment
            if close[i] < camarilla_s3[i] or ema_50_12h_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses downtrend alignment
            if close[i] > camarilla_r3[i] or ema_50_12h_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals