#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation.
# Uses daily Donchian channel (20) breakouts for trend entry.
# Choppiness Index (14) on 12h determines market regime: >61.8 = range (mean revert), <38.2 = trending (trend follow).
# Volume confirmation (current volume > 1.5x 20-period average) filters low-quality breakouts.
# In trending regime (CHOP < 38.2): enter long on Donchian high breakout, short on Donchian low breakdown.
# In ranging regime (CHOP > 61.8): enter long near Donchian low, short near Donchian high (mean reversion).
# ATR-based stoploss (2.5x ATR) manages risk.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_chop_donchian_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Choppiness Index (14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first period has no previous close
    
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[0:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    sum_tr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            sum_tr[i] = np.sum(tr[0:15])
        else:
            sum_tr[i] = sum_tr[i-1] - tr[i-14] + tr[i]
    
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # avoid division by zero
    
    # 1d Donchian channel (20-period) - HTF
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(19, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(chop[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr_val = atr[i] if not np.isnan(atr[i]) else (high[i] - low[i])
            stop_loss_level = entry_price - 2.5 * atr_val
            
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            atr_val = atr[i] if not np.isnan(atr[i]) else (high[i] - low[i])
            stop_loss_level = entry_price + 2.5 * atr_val
            
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries based on regime and volume confirmation
            if volume_filter:
                if chop[i] < 38.2:  # Trending regime
                    # Breakout above Donchian high
                    if close[i] > donchian_high_aligned[i] and close[i-1] <= donchian_high_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Breakdown below Donchian low
                    elif close[i] < donchian_low_aligned[i] and close[i-1] >= donchian_low_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                elif chop[i] > 61.8:  # Ranging regime
                    # Mean reversion: long near Donchian low
                    if close[i] <= donchian_low_aligned[i] * 1.001 and close[i-1] > donchian_low_aligned[i] * 1.001:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    # Mean reversion: short near Donchian high
                    elif close[i] >= donchian_high_aligned[i] * 0.999 and close[i-1] < donchian_high_aligned[i] * 0.999:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
    
    return signals