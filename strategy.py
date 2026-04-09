#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with volume confirmation
# Weekly Camarilla pivots provide major support/resistance levels based on prior week's range
# Long when price breaks above weekly H3 with volume confirmation
# Short when price breaks below weekly L3 with volume confirmation
# Uses discrete position sizing 0.25 to target ~12-37 trades/year (50-150 over 4 years)
# Weekly timeframe reduces noise and captures major market structure, working in both bull and bear markets

name = "6h_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.zeros_like(close_1w)
    
    # Calculate weekly Camarilla pivot levels
    # Camarilla: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    range_1w = high_1w - low_1w
    camarilla_h3 = close_1w + 1.1 * range_1w
    camarilla_l3 = close_1w - 1.1 * range_1w
    
    # Calculate weekly average volume (10-period) for confirmation
    vol_s_1w = pd.Series(volume_1w)
    avg_vol_1w = vol_s_1w.rolling(window=10, min_periods=10).mean().values
    
    # Align weekly indicators to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    avg_vol_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(avg_vol_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average weekly volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        if position == 1:  # Long position
            # Exit long if price falls below weekly L3 (stop/reversal)
            if close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above weekly H3 (stop/reversal)
            if close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on weekly H3/L3 break with volume confirmation
            if close[i] > camarilla_h3_aligned[i] and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif close[i] < camarilla_l3_aligned[i] and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals