#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter
# - Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar avg AND 1d chop < 61.8 (trending)
# - Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar avg AND 1d chop < 61.8 (trending)
# - Exit when price returns to Donchian(20) midpoint
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - Donchian breakouts work in both bull and bear markets when combined with volume and regime filters
# - Volume confirmation filters false breakouts
# - 1d chop regime ensures we only trade in trending conditions (avoids whipsaws in ranging markets)

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h Donchian(20) channels
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Pre-compute 1d chop regime: Chopiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with 1d bars
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sum(tr)/atr) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr_14 * 14)) / np.log10(14)
    
    # Align chop to 4h timeframe (completed 1d bar only)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute 4h volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending regime (chop < 61.8)
        if chop_aligned[i] >= 61.8:
            # In ranging market, exit any position
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike
            if (prices['close'].iloc[i] > high_20[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike
            elif (prices['close'].iloc[i] < low_20[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when price returns to Donchian midpoint
            if position == 1 and prices['close'].iloc[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and prices['close'].iloc[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals