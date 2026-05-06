#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d trend filter and volume confirmation
# Donchian(20) on 4h provides clear breakout levels; 1d EMA50 filters trend direction
# Volume > 1.5x 20-period average confirms momentum; session filter (08-20 UTC) reduces noise
# Targets 15-37 trades/year by using 4h for direction, 1h only for entry timing
# Works in bull/bear markets: breakouts capture trends, trend filter avoids counter-trend trades

name = "1h_Donchian20_1dEMA50_VolumeFilter_v1"
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
    
    # 4h Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = ema_50_1d > 0  # Placeholder for actual trend logic
    
    # Align 1d EMA50 trend to 1h timeframe
    # We'll use price vs EMA comparison: 1 if price > EMA (uptrend), -1 if price < EMA (downtrend)
    price_vs_ema = np.where(close_1d > ema_50_1d, 1, -1)
    trend_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(trend_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high with volume and uptrend
            if close[i] > donch_high_aligned[i] and volume_filter[i] and trend_aligned[i] == 1:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below 4h Donchian low with volume and downtrend
            elif close[i] < donch_low_aligned[i] and volume_filter[i] and trend_aligned[i] == -1:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low (failed breakout)
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high (failed breakdown)
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals