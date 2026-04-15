#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Donchian breakout with 1d volume confirmation and session filter
# Long when price breaks above 4h Donchian upper (20) + 1d volume > 1.5x 20-period average + in session (08-20 UTC)
# Short when price breaks below 4h Donchian lower (20) + 1d volume > 1.5x 20-period average + in session
# Uses 4h for signal direction (structure), 1h for entry timing precision, 1d volume filter for conviction
# Discrete position size 0.20 to minimize fee drag and control drawdown in bear markets.
# Target: 15-30 trades/year per symbol to avoid fee drag on 1h timeframe.

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
    
    # Get 4h HTF data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Donchian Channel (20) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Get 1d HTF data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Volume SMA (20) ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20) + 5  # Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period volume SMA
        # Note: we use the 1d volume value aligned to current 1h bar
        vol_confirm = (df_1d['volume'].iloc[-1] if len(df_1d) > 0 else 0) > (vol_sma_20_1d_aligned[i] * 1.5)
        # Alternative: use current 1h volume vs 1d volume average (more responsive)
        vol_confirm = volume[i] > (vol_sma_20_1d_aligned[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (close > upper)
        # 2. Volume confirmation
        if (close[i] > donchian_high_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (close < lower)
        # 2. Volume confirmation
        elif (close[i] < donchian_low_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Donchian20_1dVolumeFilter_Session_v1"
timeframe = "1h"
leverage = 1.0