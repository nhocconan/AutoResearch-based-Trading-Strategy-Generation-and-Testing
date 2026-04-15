#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Williams %R filter and volume confirmation
# Long when price breaks above 6h Donchian upper (20-period) + 1d Williams %R < -80 (oversold) + volume > 1.5x 20-period avg
# Short when price breaks below 6h Donchian lower (20-period) + 1d Williams %R > -20 (overbought) + volume > 1.5x 20-period avg
# Williams %R acts as a contrarian filter: we buy breakouts from oversold conditions and sell breakdowns from overbought conditions
# This avoids buying strength and selling weakness, improving performance in both bull and bear markets
# Target: 50-150 total trades over 4 years = 12-37/year. Position size: 0.25

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Williams %R (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.where(hl_range != 0, ((highest_high - close_1d) / hl_range) * -100, -50)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 6h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 14) + 20  # Donchian(20) + Williams %R(14) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h Donchian upper (20-period)
        # 2. 1d Williams %R < -80 (oversold)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           (williams_r_aligned[i] < -80) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h Donchian lower (20-period)
        # 2. 1d Williams %R > -20 (overbought)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             (williams_r_aligned[i] > -20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1dWilliamsR_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0