#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian breakout with volume confirmation and chop filter
# - Long when price breaks above 20-period 1d Donchian high with volume > 1.5x 20-period average
# - Short when price breaks below 20-period 1d Donchian low with volume > 1.5x 20-period average
# - Only take trades when 1d Choppiness Index > 61.8 (ranging market) for mean-reversion logic
# - Uses 12h timeframe for execution but 1d for signal generation to reduce trade frequency
# - Position size: 0.25 (25% of capital) to balance risk and reward
# - Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions

name = "12h_Donchian20_1dVol_Chop_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and Choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period 1d Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: highest high over last 20 periods
    donchian_high = np.full(len(high_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
    
    # Donchian low: lowest low over last 20 periods
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(low_1d)):
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d Choppiness Index (20-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    atr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - df_1d['close'].values[i-1]),
            abs(low_1d[i] - df_1d['close'].values[i-1])
        )
        atr_1d[i] = tr
    
    chop = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        sum_atr = np.sum(atr_1d[i-20:i])
        max_high = np.max(high_1d[i-20:i])
        min_low = np.min(low_1d[i-20:i])
        if max_high - min_low > 0:
            chop[i] = 100 * np.log10(sum_atr) / np.log10(20) / np.log10(max_high - min_low)
        else:
            chop[i] = 50  # neutral when no range
    
    # Align Choppiness to 12h timeframe
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or
            np.isnan(chop_12h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in ranging market (Choppiness > 61.8)
        if chop_12h[i] <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long when price breaks above Donchian high with volume confirmation
            if close[i] > donchian_high_12h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low with volume confirmation
            elif close[i] < donchian_low_12h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel (mean reversion)
            if close[i] < donchian_high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel (mean reversion)
            if close[i] > donchian_low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals