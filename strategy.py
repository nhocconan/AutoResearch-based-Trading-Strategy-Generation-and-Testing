#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Donchian breakouts capture momentum with clear structure. Volume confirmation ensures
# breakout strength. Choppiness filter avoids whipsaws in ranging markets (CHOP > 61.8).
# Works in both bull/bear markets by requiring volume spike and regime alignment.
# Discrete position sizing (0.25) limits drawdown and reduces fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (14-period) - measures ranging vs trending markets
    # CHOP > 61.8 = ranging (choppy), CHOP < 38.2 = trending
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_14 * 14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # Neutral when no range
    
    # Regime filter: CHOP < 61.8 (allow trading in trending/low-chop regimes)
    chop_filter = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_confirm[i]
        chop_ok = chop_filter[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian high, volume confirm, chop filter
            if price > donchian_high[i] and vol_confirm and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low, volume confirm, chop filter
            elif price < donchian_low[i] and vol_confirm and chop_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to Donchian low
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement to Donchian high
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals