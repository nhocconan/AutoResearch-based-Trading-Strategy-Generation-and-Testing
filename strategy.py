#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike filter and session time filter
# Long when price breaks above 4h Donchian upper + 12h volume > 2.0x 20-period avg + UTC 08-20
# Short when price breaks below 4h Donchian lower + 12h volume > 2.0x 20-period avg + UTC 08-20
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# Donchian channels provide objective breakout levels. Volume spike confirms institutional interest.
# Session filter avoids low-liquidity overnight hours. Works in bull markets (trend continuation) 
# and bear markets (strong downtrends) by requiring volume confirmation on breakouts.

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
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Volume (for spike detection) ===
    vol_12h = df_12h['volume'].values
    vol_sma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 20) + 5  # Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 2.0x 20-period volume SMA
        vol_spike = volume[i] > (vol_12h_aligned[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Volume spike confirmation (12h)
        if (close[i] > donchian_high[i]) and vol_spike:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Volume spike confirmation (12h)
        elif (close[i] < donchian_low[i]) and vol_spike:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hVolSpike2x_Session_Filter_v1"
timeframe = "4h"
leverage = 1.0