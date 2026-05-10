#!/usr/bin/env python3
# 4H_1D_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla R3/S3 levels from 1d act as strong reversal points when confirmed by daily trend and volume spikes.
# Long when price breaks above R3 in a daily uptrend with volume spike.
# Short when price breaks below S3 in a daily downtrend with volume spike.
# Uses 1d EMA34 for trend filter and volume > 1.5x 20-period average for confirmation.
# Works in bull/bear by following daily trend direction. Target: 20-40 trades/year per symbol.

name = "4H_1D_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Calculate Camarilla levels for 1d
    # R4 = close + 1.5*(high - low)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # S4 = close - 1.5*(high - low)
    hl_range = high_1d - low_1d
    r3 = close_1d + 1.1 * hl_range
    s3 = close_1d - 1.1 * hl_range
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Trend: bullish if close > EMA34, bearish if close < EMA34
    bullish_trend = close_1d > ema34_1d
    bearish_trend = close_1d < ema34_1d
    
    # Volume spike: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma20
    
    # Align to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        vol_spike = volume_spike_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: price breaks above R3, bullish trend, volume spike
            if high[i] > r3_aligned[i] and bullish and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3, bearish trend, volume spike
            elif low[i] < s3_aligned[i] and bearish and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 or bearish trend
            if low[i] < s3_aligned[i] or bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 or bullish trend
            if high[i] > r3_aligned[i] or bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals