#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction and 1d Donchian breakout for timing
# - 4h Supertrend(10,3) determines bull/bear regime (works in both bull/bear markets)
# - 1d Donchian(20) breakout in direction of 4h trend captures momentum with structure
# - Session filter (08-20 UTC) reduces noise and whipsaw trades
# - Position size: 0.20 (discrete level to minimize fee churn)
# - Target: 15-37 trades/year (60-150 total over 4 years) for 1h timeframe
# - Novelty: Combines reliable 4h trend filter with 1d breakout precision to avoid false signals

name = "1h_4h_1d_supertrend_donchian_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h Supertrend
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr_4h[0]
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + (3.0 * atr_4h)
    lower_band_4h = hl2_4h - (3.0 * atr_4h)
    
    supertrend_4h = np.full_like(close_4h, np.nan, dtype=float)
    direction_4h = np.ones_like(close_4h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i-1] > supertrend_4h[i-1]:
            upper_band_4h[i] = min(upper_band_4h[i], upper_band_4h[i-1])
        else:
            lower_band_4h[i] = max(lower_band_4h[i], lower_band_4h[i-1])
        
        if close_4h[i] > upper_band_4h[i-1]:
            direction_4h[i] = 1
        elif close_4h[i] < lower_band_4h[i-1]:
            direction_4h[i] = -1
        else:
            direction_4h[i] = direction_4h[i-1]
        
        supertrend_4h[i] = lower_band_4h[i] if direction_4h[i] == 1 else upper_band_4h[i]
    
    # Align 4h Supertrend direction to 1h timeframe
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h.astype(float))
    
    # Pre-compute 1d Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 1h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # 1h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(supertrend_4h_aligned[i]) or 
            np.isnan(direction_4h_aligned[i]) or
            np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 1d Donchian low OR 4h trend turns bearish
            if low[i] <= donchian_low_20_aligned[i] or direction_4h_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above 1d Donchian high OR 4h trend turns bullish
            if high[i] >= donchian_high_20_aligned[i] or direction_4h_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price breaks above 1d Donchian high AND 4h trend is bullish
            if high[i] >= donchian_high_20_aligned[i] and direction_4h_aligned[i] == 1:
                position = 1
                signals[i] = 0.20
            # Enter short: price breaks below 1d Donchian low AND 4h trend is bearish
            elif low[i] <= donchian_low_20_aligned[i] and direction_4h_aligned[i] == -1:
                position = -1
                signals[i] = -0.20
    
    return signals