#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h Supertrend(ATR=10,mult=3) trend filter and volume confirmation
# Long when price breaks above Donchian upper + 12h Supertrend uptrend + volume > 2.0x 20-period avg
# Short when price breaks below Donchian lower + 12h Supertrend downtrend + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 12h Supertrend provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (2.0x) targets ~12-37 trades/year on 6h timeframe to avoid overtrading.
# Donchian channels calculated from prior 6h bar's high/low for structure-based entries.

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ATR calculation
        return np.zeros(n)
    
    # === 12h Indicator: Supertrend(ATR=10,mult=3) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(10)
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_10[i-1]) or np.isnan(close_12h[i-1]):
            continue
            
        # Upper band calculation
        if close_12h[i-1] <= supertrend[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        # Lower band calculation
        if close_12h[i-1] >= supertrend[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
        
        # Supertrend calculation
        if close_12h[i] <= upper_band[i]:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            supertrend[i] = lower_band[i]
            direction[i] = 1
    
    # Align Supertrend direction to 6h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction.astype(float))
    
    # === 6h Donchian Channel (20) ===
    # Using prior bar's high/low to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    donchian_upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 5  # Supertrend(30) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(supertrend_direction_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. 12h Supertrend uptrend (direction == 1)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and \
           (supertrend_direction_aligned[i] == 1) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. 12h Supertrend downtrend (direction == -1)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and \
             (supertrend_direction_aligned[i] == -1) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_12hSupertrend_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0