#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Choppiness Index regime filter + Donchian breakout + volume confirmation.
# In trending regime (CHOP < 38.2): breakout above/below Donchian(20) with volume > 1.5x average.
# In ranging regime (CHOP > 61.8): mean reversion at Donchian bands with volume confirmation.
# Uses regime filter to avoid whipsaws in sideways markets, targeting 20-40 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate ATR(14) for 1d
    atr_period = 14
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(tr)):
        if i == atr_period:
            atr_1d[i] = np.nanmean(tr[1:i+1])  # Skip first NaN
        else:
            atr_1d[i] = (atr_1d[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate Choppiness Index(14)
    chop_period = 14
    chop = np.full(len(close_1d), np.nan)
    for i in range(chop_period, len(close_1d)):
        if np.isnan(atr_1d[i-chop_period+1:i+1]).any():
            continue
        atr_sum = np.nansum(atr_1d[i-chop_period+1:i+1])
        highest_high = np.nanmax(high_1d[i-chop_period+1:i+1])
        lowest_low = np.nanmin(low_1d[i-chop_period+1:i+1])
        if highest_high == lowest_low:
            chop[i] = 50.0  # Avoid division by zero
        else:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(chop_period)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels (20-period)
    donch_period = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        donch_high[i] = np.max(high[i - donch_period + 1:i + 1])
        donch_low[i] = np.min(low[i - donch_period + 1:i + 1])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, Chop, and volume MA20
    start_idx = max(donch_period - 1, chop_period + atr_period, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            if chop_val < 38.2:  # Trending regime
                # Long: break above Donchian high with volume filter
                if price > donch_high[i] and vol_filter:
                    signals[i] = size
                    position = 1
                # Short: break below Donchian low with volume filter
                elif price < donch_low[i] and vol_filter:
                    signals[i] = -size
                    position = -1
            elif chop_val > 61.8:  # Ranging regime
                # Long: bounce off Donchian low with volume filter
                if price < donch_low[i] * 1.005 and vol_filter:  # Near lower band
                    signals[i] = size
                    position = 1
                # Short: bounce off Donchian high with volume filter
                elif price > donch_high[i] * 0.995 and vol_filter:  # Near upper band
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit conditions
            if chop_val < 38.2:  # Trending: exit at midline
                if price < donch_mid[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Ranging: exit at opposite band or midline
                if price > donch_high[i] * 0.995 or price < donch_mid[i]:
                    signals[i] = 0.0
                    position = 0
            if position == 1:
                signals[i] = size
        elif position == -1:
            # Exit conditions
            if chop_val < 38.2:  # Trending: exit at midline
                if price > donch_mid[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Ranging: exit at opposite band or midline
                if price < donch_low[i] * 1.005 or price > donch_mid[i]:
                    signals[i] = 0.0
                    position = 0
            if position == -1:
                signals[i] = -size
    
    return signals

name = "4h_ChopRegime_DonchianBreakout_Volume"
timeframe = "4h"
leverage = 1.0