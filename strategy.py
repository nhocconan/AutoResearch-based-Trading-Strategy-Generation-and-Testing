#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h trend filter and volume confirmation
# Uses Donchian channel breakouts (20-period) for breakout entries, filtered by
# 12h EMA50 trend direction and volume spike confirmation. Works in both bull
# and bear markets by trading breakouts in the direction of the higher timeframe
# trend. Volume ensures momentum behind breakouts. Target: 15-30 trades/year.

name = "6h_DonchianBreakout_12hTrend_VolumeConfirm"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg[i] = np.mean(volume[i-19:i+1])
    
    # Align 12h EMA50 to 6h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_confirm = vol_current > 1.5 * vol_avg[i]  # Volume spike
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume
            if price > donchian_high[i] and price > ema50_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume
            elif price < donchian_low[i] and price < ema50_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend changes
            if price < donchian_low[i] or price < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend changes
            if price > donchian_high[i] or price > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals