#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation
# Uses 4h Donchian(20) for structure, 1d EMA50 for trend alignment (works in bull/bear), volume spike (>2x 24-bar avg) for confirmation
# Session filter (08-20 UTC) reduces noise. Discrete sizing 0.20 to limit fee drag. Target 80-160 total trades (20-40/year)
# Proven pattern: HTF structure + volume confirmation + trend filter works on BTC/ETH in all regimes

name = "1h_Donchian20_1dEMA50_VolumeConfirm_v1"
timeframe = "1h"
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_1d = df_1d['close'].values
    volume_1h = volume  # use primary timeframe volume
    
    # Calculate 4h Donchian channel (20-period)
    high_ma_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike filter (>2.0x 24-bar average on 1h)
    vol_ma_24 = pd.Series(volume_1h).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume_1h > (2.0 * vol_ma_24)
    
    # Align HTF indicators to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, high_ma_20)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, low_ma_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)  # align 1h volume filter using 1d df for proper indexing
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper 4h Donchian AND uptrend (price > 1d EMA50) AND volume spike
            if close[i] > upper_4h_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < lower 4h Donchian AND downtrend (price < 1d EMA50) AND volume spike
            elif close[i] < lower_4h_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retests lower 4h Donchian from above
            if close[i] <= lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retests upper 4h Donchian from below
            if close[i] >= upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals