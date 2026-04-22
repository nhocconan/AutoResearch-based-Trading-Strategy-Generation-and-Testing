#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian breakout with volume confirmation
# Uses Choppiness Index (14) to distinguish trending vs ranging markets.
# In trending markets (CHOP < 38.2): trade Donchian(20) breakouts with volume confirmation.
# In ranging markets (CHOP > 61.8): fade Donchian breakouts (mean reversion).
# Adaptive approach works in both bull/bear markets by adjusting to current regime.
# Designed for 4h timeframe with tight entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Choppiness Index calculation (trend regime filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range for Choppiness Index
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_12h
    
    # Calculate ATR(14) for Choppiness Index
    atr_14 = np.zeros_like(close_12h)
    atr_14[:] = np.nan
    for i in range(14, len(close_12h)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR14) / (max(high) - min(low)) * 1/14) / log10(14)
    chop = np.full_like(close_12h, np.nan)
    for i in range(27, len(close_12h)):  # need 14+14 bars for calculation
        sum_atr = np.nansum(atr_14[i-13:i+1])
        max_high = np.nanmax(high_12h[i-13:i+1])
        min_low = np.nanmin(low_12h[i-13:i+1])
        if max_high > min_low and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / (max_high - min_low) * 1/14) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Donchian Channel (20) on 4h data
    donchian_high = np.zeros_like(close)
    donchian_low = np.zeros_like(close)
    donchian_high[:] = np.nan
    donchian_low[:] = np.nan
    for i in range(20, len(close)):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Trending market regime (CHOP < 38.2): trade breakouts
            if chop_val < 38.2:
                # Long breakout with volume confirmation
                if (close[i] > donchian_high[i] and 
                    volume[i] > 1.5 * vol_avg_20[i]):
                    signals[i] = 0.25
                    position = 1
                # Short breakdown with volume confirmation
                elif (close[i] < donchian_low[i] and 
                      volume[i] > 1.5 * vol_avg_20[i]):
                    signals[i] = -0.25
                    position = -1
            # Ranging market regime (CHOP > 61.8): fade breakouts (mean reversion)
            elif chop_val > 61.8:
                # Short at resistance (sell breakdown hope)
                if (close[i] > donchian_high[i] and 
                    volume[i] > 1.5 * vol_avg_20[i]):
                    signals[i] = -0.20
                    position = -1
                # Long at support (buy breakdown fear)
                elif (close[i] < donchian_low[i] and 
                      volume[i] > 1.5 * vol_avg_20[i]):
                    signals[i] = 0.20
                    position = 1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Donchian breakdown OR regime shift to ranging
                if (close[i] < donchian_low[i] or chop_val > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Donchian breakout OR regime shift to ranging
                if (close[i] > donchian_high[i] or chop_val > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "4h_Chop_Donchian_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0