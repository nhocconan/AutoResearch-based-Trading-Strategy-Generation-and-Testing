#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Camarilla pivot levels + volume confirmation + chop regime filter
# Strategy: Long when price touches Camarilla L3 with volume > 1.5x avg and CHOP > 50 (range)
# Short when price touches Camarilla H3 with volume > 1.5x avg and CHOP > 50 (range)
# Uses 1d Camarilla levels for mean reversion in ranging markets, volume confirms rejection
# Chop filter ensures we only trade in ranging conditions (avoid trending markets)
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = close + 1.1*(high-low)*1.1/2
    # L3 = close - 1.1*(high-low)*1.1/2
    # We'll use the previous day's values to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first period uses current values
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    # Calculate Choppiness Index (14) on 1d for regime detection
    # CHOP = 100 * log10(sum(TR)/ (ATR * n)) / log10(n)
    # Simplified: CHOP = 100 * log10(sum of TR over n periods) / log10(n) - 100 * log10(ATR * n) / log10(n)
    # We'll use the standard formula: 100 * log10(sumTR / (atr * n)) / log10(n)
    
    # True Range
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.concatenate([[tr3[0]] if len(tr3) > 0 else [0], tr3])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # ATR (14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Chop = 100 * log10(sum_tr_14 / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14 + 1e-10)) / np.log10(14)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # Price near Camarilla levels (within 0.1% tolerance)
        tolerance = 0.001  # 0.1%
        near_h3 = abs(close[i] - camarilla_h3_aligned[i]) / camarilla_h3_aligned[i] < tolerance
        near_l3 = abs(close[i] - camarilla_l3_aligned[i]) / camarilla_l3_aligned[i] < tolerance
        
        # Range condition (choppy market)
        ranging = chop_aligned[i] > 50
        
        # Entry logic
        long_entry = near_l3 and volume_surge and ranging
        short_entry = near_h3 and volume_surge and ranging
        
        # Exit conditions: opposite touch or trend develops
        exit_long = position == 1 and (near_h3 or chop_aligned[i] < 40)
        exit_short = position == -1 and (near_l3 or chop_aligned[i] < 40)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0