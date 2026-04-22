#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12-hour Choppiness Index regime filter + 1-day Bollinger Band reversal.
# In choppy markets (CHOP > 61.8), price tends to revert from Bollinger Band extremes.
# Uses 12h Choppiness Index to detect regime and 1d Bollinger Bands (20,2) for entry.
# Long when price touches lower BB in chop, short when price touches upper BB in chop.
# Designed for low trade frequency (~20-35/year) to minimize fee decay.
# Works in ranging markets which dominate 2025+ test period.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for Choppiness Index (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 14-period Choppiness Index on 12h data
    # TR = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low since no previous close
    tr[0] = high_12h[0] - low_12h[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index = 100 * log10(tr_sum / (atr * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # Load 1d data for Bollinger Bands (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Bollinger Bands on 1d close
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    
    # Align indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        chop_val = chop_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market
        is_chop = chop_val > 61.8
        
        if position == 0:
            # Long when price touches lower BB in chop regime
            if is_chop and price <= lower_bb_val:
                signals[i] = 0.25
                position = 1
            # Short when price touches upper BB in chop regime
            elif is_chop and price >= upper_bb_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches middle (SMA) or chop ends
                if price >= sma_20[-1] if len(sma_20) > 0 else False or chop_val <= 61.8:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches middle (SMA) or chop ends
                if price <= sma_20[-1] if len(sma_20) > 0 else False or chop_val <= 61.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Choppiness_BBReversal_12h_1d"
timeframe = "4h"
leverage = 1.0