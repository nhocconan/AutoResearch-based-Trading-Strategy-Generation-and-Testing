#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction and 1d Donchian breakout for entry timing.
# Long when: 4h Supertrend = uptrend, price breaks above 1d Donchian upper (20), volume > 1.5x 20-period average, and within 08-20 UTC session.
# Short when: 4h Supertrend = downtrend, price breaks below 1d Donchian lower (20), volume > 1.5x 20-period average, and within 08-20 UTC session.
# Uses discrete position sizing (0.20) to minimize fee churn. Designed for low trade frequency (15-35/year).
# Supertrend filters choppy markets; Donchian breakouts capture momentum. Volume confirmation avoids false breakouts.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by following 4h trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop for Supertrend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Supertrend (ATR=10, mult=3.0) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = high_4h[0] - low_4h[0]
    tr2[0] = np.abs(high_4h[0] - close_4h[0])
    tr3[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr_period = 10
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + (3.0 * atr)
    lower = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upper[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lower[i]
        elif direction[i] == 1:
            supertrend[i] = max(upper[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower[i], supertrend[i-1])
    
    # 1 if uptrend, -1 if downtrend
    supertrend_direction = direction
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, supertrend_direction)
    
    # === 1d Indicator: Donchian Channels (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 4h Supertrend = uptrend (1)
        # 2. Price breaks above 1d Donchian upper (20)
        # 3. Volume confirmation
        if (supertrend_direction_aligned[i] == 1) and \
           (close[i] > donchian_upper_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 4h Supertrend = downtrend (-1)
        # 2. Price breaks below 1d Donchian lower (20)
        # 3. Volume confirmation
        elif (supertrend_direction_aligned[i] == -1) and \
             (close[i] < donchian_lower_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Supertrend4h_Donchian1d_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0