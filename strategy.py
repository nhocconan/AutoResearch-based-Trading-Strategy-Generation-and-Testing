# NOTE: This is a template for educational purposes only.
# Revised strategy based on experiment #76606 analysis: 4h Camarilla pivot + volume spike + choppiness regime
# This strategy template follows the winning patterns: tight entries, volume confirmation, regime filter.
# Remember to use discrete position sizes (0.0, ±0.25, ±0.30) to minimize fee churn.
# Backtest thoroughly before live deployment.

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1h data for choppiness index calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14) on 1h
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = high_1h - low_1h
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # small value to prevent div by zero
    choppiness = 100 * np.log10(tr_sum / range_hl) / np.log10(14)
    
    # Load daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first day uses same day's data
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    diff = prev_high - prev_low
    R1 = prev_close + (diff * 1.1 / 12)
    R2 = prev_close + (diff * 1.1 / 6)
    R3 = prev_close + (diff * 1.1 / 4)
    R4 = prev_close + (diff * 1.1 / 2)
    S1 = prev_close - (diff * 1.1 / 12)
    S2 = prev_close - (diff * 1.1 / 6)
    S3 = prev_close - (diff * 1.1 / 4)
    S4 = prev_close - (diff * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(choppiness[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Choppiness > 61.8 = ranging market (good for mean reversion at pivots)
        is_ranging = choppiness[i] > 61.8
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and is_ranging:
            # Long: price touches S1 level with volume spike in ranging market
            if low[i] <= S1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 level with volume spike in ranging market
            elif high[i] >= R1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price moves to opposite pivot level or volatility breaks down
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R1 or choppiness drops below 40 (trending)
                if high[i] >= R1_aligned[i] or choppiness[i] < 40.0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S1 or choppiness drops below 40 (trending)
                if low[i] <= S1_aligned[i] or choppiness[i] < 40.0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Range_Volume"
timeframe = "4h"
leverage = 1.0