#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + choppiness regime filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg AND chop > 61.8 (range regime)
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg AND chop > 61.8 (range regime)
# Exit when price crosses Donchian(10) mid-line (5-bar Donchian midpoint)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-40 trades/year on 4h.
# Works in bull/bear via range reversion in choppy markets (chop > 61.8) and volume-confirmed breakouts
# Donchian channels provide objective price levels; chop filter ensures mean-reversion context

name = "4h_Donchian20_Volume_Chop_Reversion_v1"
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
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14) sum
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    chop = 100 * np.log10(atr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian channels on 4h
    donch_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donch_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donch_10_mid = (donch_10_high + donch_10_low) / 2
    
    # Volume confirmation: >1.5x 20-bar average
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_20_high[i]) or np.isnan(donch_20_low[i]) or 
            np.isnan(donch_10_mid[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        chop_val = chop_aligned[i]
        curr_close = close[i]
        
        # Only trade in choppy/range regimes (chop > 61.8)
        if chop_val > 61.8:
            if position == 0:  # Flat - look for new entries
                # Long when price breaks above Donchian(20) high AND volume confirmation
                if curr_close > donch_20_high[i] and vol_conf:
                    signals[i] = 0.25
                    position = 1
                # Short when price breaks below Donchian(20) low AND volume confirmation
                elif curr_close < donch_20_low[i] and vol_conf:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:  # Long - exit when price crosses Donchian(10) mid-line
                if curr_close < donch_10_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short - exit when price crosses Donchian(10) mid-line
                if curr_close > donch_10_mid[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending regimes, stay flat
            signals[i] = 0.0
    
    return signals