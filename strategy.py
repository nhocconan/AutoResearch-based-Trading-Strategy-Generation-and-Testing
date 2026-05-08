#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend and volume confirmation
# Long when price breaks above R3 (bullish breakout) with 1d uptrend and volume spike
# Short when price breaks below S3 (bearish breakdown) with 1d downtrend and volume spike
# Uses Camarilla levels from 1d, 1d EMA for trend, and 60-period volume average for spike confirmation
# Designed to work in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d average volume (60-period) for volume spike filter
    daily_volume = df_1d['volume'].values
    avg_volume_60 = pd.Series(daily_volume).rolling(window=60, min_periods=60).mean().values
    avg_volume_60_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_60)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    prev_daily_close = df_1d['close'].shift(1).values
    prev_daily_high = df_1d['high'].shift(1).values
    prev_daily_low = df_1d['low'].shift(1).values
    camarilla_r3 = prev_daily_close + 1.1 * (prev_daily_high - prev_daily_low)
    camarilla_s3 = prev_daily_close - 1.1 * (prev_daily_high - prev_daily_low)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_60_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        avg_volume_60_val = avg_volume_60_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        current_volume = volume[i]
        current_close = close[i]
        
        # Volume spike condition: current volume > 1.5 * 60-day average
        volume_spike = current_volume > 1.5 * avg_volume_60_val
        
        if position == 0:
            # Enter long: price breaks above R3, 1d uptrend, volume spike
            if current_close > camarilla_r3_val and ema34_1d_val > 0 and volume_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, 1d downtrend, volume spike
            elif current_close < camarilla_s3_val and ema34_1d_val < 0 and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or 1d trend turns down
            if current_close < camarilla_s3_val or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or 1d trend turns up
            if current_close > camarilla_r3_val or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals