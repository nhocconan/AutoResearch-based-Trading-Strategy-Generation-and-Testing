#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
# Hypothesis: At 6h timeframe, breakouts of Camarilla R3/S3 levels in the direction of daily trend
# with volume confirmation capture institutional flow during breakouts. Works in bull/bear by
# following daily trend direction. Volume spike filters false breakouts. Target: 12-30 trades/year.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # Using previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day will have NaN due to roll
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (camarilla_range * 1.1 / 4)
    s3 = prev_close - (camarilla_range * 1.1 / 4)
    r4 = prev_close + (camarilla_range * 1.1 / 2)
    s4 = prev_close - (camarilla_range * 1.1 / 2)
    
    # Daily trend filter: EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    # Align to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    daily_uptrend_6h = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_6h = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    volume_spike_6h = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(daily_uptrend_6h[i]) or np.isnan(daily_downtrend_6h[i]) or np.isnan(volume_spike_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, daily uptrend, volume spike
            if (high[i] > r3_6h[i] and 
                daily_uptrend_6h[i] > 0.5 and 
                volume_spike_6h[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, daily downtrend, volume spike
            elif (low[i] < s3_6h[i] and 
                  daily_downtrend_6h[i] > 0.5 and 
                  volume_spike_6h[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or daily trend turns bearish
            if (low[i] < s3_6h[i] or daily_downtrend_6h[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or daily trend turns bullish
            if (high[i] > r3_6h[i] or daily_uptrend_6h[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals