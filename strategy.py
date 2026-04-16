#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Primary timeframe: daily
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # HTF: weekly
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly volume average (20-period) for volume confirmation
    vol_ma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = volume_1w / vol_ma20_1w
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    # Daily ATR (14) for position sizing and stop reference
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily Donchian channel (20) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_20
    donchian_lower = lowest_20
    
    signals = np.zeros(n)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema50 = ema50_1w_aligned[i]
        vol_ratio = vol_ratio_1w_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            if price < donchian_lower[i]:  # Break below lower Donchian
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price > donchian_upper[i]:  # Break above upper Donchian
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # LONG: Price breaks above upper Donchian + above weekly EMA50 + volume surge
            if (price > donchian_upper[i]) and (price > ema50) and (vol_ratio > 2.0):
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price breaks below lower Donchian + below weekly EMA50 + volume surge
            elif (price < donchian_lower[i]) and (price < ema50) and (vol_ratio > 2.0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA50_DonchianBreakout_VolumeSurge"
timeframe = "1d"
leverage = 1.0