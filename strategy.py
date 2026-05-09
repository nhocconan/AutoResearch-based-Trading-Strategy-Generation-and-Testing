#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Bollinger Band reversal
# In choppy markets (CHOP > 61.8), price tends to revert to mean from Bollinger Bands
# In trending markets (CHOP < 38.2), avoid trades to prevent whipsaw
# Works in both bull/bear as regime adapts to market conditions
name = "4h_Choppiness_BB_Reversal"
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
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Bollinger Bands on 1d close (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 4h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate Choppiness Index on 4h data (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 20)  # BB, CHOP, volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or
            np.isnan(chop[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ch = chop[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Enter long: price touches lower BB in choppy market + volume
            if ch > 61.8 and close[i] <= lower and vol_filt:
                signals[i] = 0.25
                position = 1
            # Enter short: price touches upper BB in choppy market + volume
            elif ch > 61.8 and close[i] >= upper and vol_filt:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reaches middle (SMA) or chop drops (trend emerging)
            # We don't have SMA aligned, so exit when price reaches upper BB or chop < 50
            if close[i] >= upper or ch < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches middle or chop drops
            if close[i] <= lower or ch < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals