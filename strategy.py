#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 4h Donchian(20) breakout with volume confirmation
# Long when CHOP > 61.8 (range) + price breaks above Donchian high(20) + volume > 1.5x avg
# Short when CHOP > 61.8 (range) + price breaks below Donchian low(20) + volume > 1.5x avg
# Exit when CHOP < 38.2 (trend) or opposite Donchian break
# Uses 4h for all signals, targeting 20-50 trades/year for low fee drag
# Chop filter reduces whipsaws in strong trends, focusing on mean-reversion in ranges

name = "4h_ChopDonchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for Chop calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate True Range components for Chop denominator
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high - min_low)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range_hl = max_high - min_low
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Chop and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop[i]
        high_val = high[i]
        low_val = low[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: range market (CHOP > 61.8) + break above Donchian high + volume spike
            if chop_val > 61.8 and high_val > donchian_high_val and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: range market (CHOP > 61.8) + break below Donchian low + volume spike
            elif chop_val > 61.8 and low_val < donchian_low_val and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend market (CHOP < 38.2) or price breaks below Donchian low
            if chop_val < 38.2 or low_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend market (CHOP < 38.2) or price breaks above Donchian high
            if chop_val < 38.2 or high_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals