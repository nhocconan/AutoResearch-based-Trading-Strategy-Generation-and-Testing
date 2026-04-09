#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and low volatility filter
# Uses Camarilla pivot levels (L3, H3) from daily chart for mean reversion entries
# Enters long at L3 bounce with volume spike, short at H3 rejection with volume spike
# Only trades when 1d ATR rank < 40 (low volatility environment) to avoid choppy markets
# Exits at opposite Camarilla level (H3 for longs, L3 for shorts) or opposite pivot
# Position size 0.25 to limit drawdown
# Target: 25-40 trades/year per symbol to minimize fee drag while capturing reversals

name = "4h_1d_camarilla_reversal_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:
            continue  # Skip first day (no previous day data)
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
        camarilla_h4[i] = prev_close + range_val * 1.1 / 2
        camarilla_l4[i] = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (only use completed daily bars)
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d ATR (14-period)
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    atr_1d = np.zeros(len(df_1d))
    atr_1d[0] = tr_1d[0]
    for i in range(1, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # ATR percentile rank (200-day lookback)
    atr_rank_1d = np.zeros(len(df_1d))
    for i in range(200, len(df_1d)):
        window = atr_1d[i-200:i]
        atr_rank_1d[i] = np.sum(window < atr_1d[i]) / len(window) * 100
    
    # Align ATR rank to 4h timeframe (only use completed daily bars)
    atr_rank_4h = align_htf_to_ltf(prices, df_1d, atr_rank_1d)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after ATR rank warmup
        # Skip if any required data is invalid
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(atr_rank_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low volatility environment (ATR rank < 40 = bottom 40% volatility)
        if atr_rank_4h[i] >= 40:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 level or closes above H4
            if close[i] >= h3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 level or closes below L4
            if close[i] <= l3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            
            # Enter long: price at L3 level with volume spike (bounce)
            if (abs(close[i] - l3_4h[i]) < (h4_4h[i] - l4_4h[i]) * 0.02 and  # Within 2% of H-L range
                vol_ratio > 1.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price at H3 level with volume spike (rejection)
            elif (abs(close[i] - h3_4h[i]) < (h4_4h[i] - l4_4h[i]) * 0.02 and  # Within 2% of H-L range
                  vol_ratio > 1.8):
                position = -1
                signals[i] = -0.25
    
    return signals