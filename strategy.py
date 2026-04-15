#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout for direction and 1h volume spike for timing.
# Long when price breaks above 4h Donchian upper band + volume > 1.5x 20-period 1h volume SMA + session filter (08-20 UTC)
# Short when price breaks below 4h Donchian lower band + volume > 1.5x 20-period 1h volume SMA + session filter (08-20 UTC)
# Uses discrete position sizing (0.20) to control drawdown and minimize fee drag.
# 4h Donchian provides structure-based breakout levels that work in ranging and trending markets.
# Volume spike confirms momentum behind the breakout.
# Session filter reduces noise during low-volume hours.
# Target: 15-37 trades/year on 1h timeframe to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian bands to 1h timeframe (completed 4h bars only)
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # === 1h Indicator: Volume SMA (20-period) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20) + 5  # Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper band
        # 2. Volume confirmation
        if (close[i] > donchian_upper_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower band
        # 2. Volume confirmation
        elif (close[i] < donchian_lower_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Donchian20_4h_VolumeSpike_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0