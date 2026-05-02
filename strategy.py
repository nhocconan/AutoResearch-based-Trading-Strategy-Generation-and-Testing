#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and chop regime filter
# Donchian breakouts capture institutional momentum in both bull/bear markets.
# Volume spike confirms participation, chop filter avoids whipsaws in ranging markets.
# Target: 75-200 trades over 4 years (19-50/year) on BTC/ETH/SOL.

name = "4h_Donchian20_Breakout_Volume_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chopiness Index (14) - range: 0-100, >61.8 = ranging, <38.2 = trending
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = np.where(
        (hh - ll) > 0,
        100 * np.log10(atr_sum / (hh - ll)) / np.log10(14),
        50  # neutral when range=0
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian(20) channels
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(donchian_window, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending regime (chop < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian with volume spike AND trending regime
            if (close[i] > upper_channel[i] and 
                volume_spike[i] and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian with volume spike AND trending regime
            elif (close[i] < lower_channel[i] and 
                  volume_spike[i] and 
                  is_trending):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below lower Donchian OR chop increases to ranging
            if close[i] < lower_channel[i] or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian OR chop increases to ranging
            if close[i] > upper_channel[i] or chop_aligned[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals