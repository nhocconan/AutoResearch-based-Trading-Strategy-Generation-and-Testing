#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band mean reversion with 1w trend filter and volume confirmation
# Uses Bollinger Bands (20, 2.0) on 12h timeframe for mean reversion signals
# Takes long when price touches lower band, short when price touches upper band
# Requires 1w EMA(50) trend filter to avoid counter-trend trades in strong trends
# Volume confirmation (>1.8x 20-bar average) ensures participation
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in both bull/bear: mean reversion works in ranges, trend filter avoids whipsaws in trends

name = "12h_BollingerMeanRev_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_12h) < 30 or len(df_1w) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands on 12h timeframe (20, 2.0)
    ma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_band = ma_20 + (2.0 * std_20)
    lower_band = ma_20 - (2.0 * std_20)
    
    # Calculate 1w EMA(50) trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume confirmation filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Align HTF indicators to 12h timeframe (primary)
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price touches lower Bollinger Band AND price above 1w EMA(50) (uptrend filter) AND volume confirmation
            if (close[i] <= lower_band_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price touches upper Bollinger Band AND price below 1w EMA(50) (downtrend filter) AND volume confirmation
            elif (close[i] >= upper_band_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches or crosses above middle Bollinger Band (20-period SMA)
            if close[i] >= ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches or crosses below middle Bollinger Band (20-period SMA)
            if close[i] <= ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals