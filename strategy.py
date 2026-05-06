#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d trend filter
# 4h Donchian(20) provides robust trend signals with clear breakout levels
# Volume > 1.8x 20-period average confirms institutional participation
# 1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, filter avoids whipsaws
# Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing

name = "1h_Donchian20_4hVol_1dTrend_v1"
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
    
    # Calculate 4h Donchian channels ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Donchian(20) - upper and lower bands
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    dh_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(dh_4h_aligned[i]) or np.isnan(dl_4h_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 4h Donchian high with volume confirmation and 1d uptrend
            if close[i] > dh_4h_aligned[i] and volume_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout: price breaks below 4h Donchian low with volume confirmation and 1d downtrend
            elif close[i] < dl_4h_aligned[i] and volume_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low (trend reversal)
            if close[i] < dl_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high (trend reversal)
            if close[i] > dh_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals