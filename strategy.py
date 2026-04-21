#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index (CI) regime filter + Donchian breakout with volume confirmation.
# Long when CI > 61.8 (range) and price breaks above Donchian(20) high with volume > 1.5x avg.
# Short when CI > 61.8 (range) and price breaks below Donchian(20) low with volume > 1.5x avg.
# Uses 1d Choppiness Index to avoid trending markets where breakouts fail.
# Target: 15-40 trades/year by requiring range regime + breakout + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d for Choppiness Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Sum of TR over 14 periods
    atr_sum = np.nansum(tr.reshape(-1, 14), axis=1) if len(tr) >= 14 else np.full(len(tr), np.nan)
    atr_sum = np.concatenate([np.full(13, np.nan), atr_sum])  # Align with original index
    
    # Highest high and lowest low over 14 periods
    hh = np.maximum.accumulate(high_1d)
    ll = np.minimum.accumulate(low_1d)
    hh_14 = np.concatenate([np.full(13, np.nan), hh[13:]])
    ll_14 = np.concatenate([np.full(13, np.nan), ll[13:]])
    
    # Chop calculation: 100 * log15(ATR_sum / (HH - LL)) / log15(14)
    # Where log15(x) = log(x) / log(15)
    hh_ll = hh_14 - ll_14
    chop_raw = 100 * (np.log(at_r_sum) - np.log(hh_ll)) / np.log(15) / np.log(14) if np.log(15) != 0 else np.full(len(close_1d), np.nan)
    chop = np.where(hh_ll > 0, 100 * np.log10(at_r_sum / hh_ll) / np.log10(14), 50.0)  # Simplified: log10(x)/log10(14) = log14(x)
    chop = np.where(hh_ll > 0, 100 * np.log(at_r_sum / hh_ll) / np.log(14), 50.0)
    
    # Handle edge cases
    chop = np.where((hh_ll > 0) & (~np.isnan(at_r_sum)) & (~np.isnan(hh_ll)), 100 * np.log(at_r_sum / hh_ll) / np.log(14), 50.0)
    
    # Align 1d Chop to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(chop_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range regime filter: Chop > 61.8 indicates ranging market
        range_regime = chop_1d_aligned[i] > 61.8
        
        if not range_regime:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        high_window = prices['high'].iloc[lookback_start:i+1].values
        low_window = prices['low'].iloc[lookback_start:i+1].values
        
        donchian_high = np.max(high_window)
        donchian_low = np.min(low_window)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period volume average
        vol_lookback_start = max(0, i - 19)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_20 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20
        
        if position == 0:
            # Enter long on breakout above Donchian high with volume in ranging market
            if price > donchian_high and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short on breakout below Donchian low with volume in ranging market
            elif price < donchian_low and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Donchian opposite level
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low
                if price < donchian_low:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high
                if price > donchian_high:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Chop_Range_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0