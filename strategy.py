#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Breakout_Volume_Regime
Hypothesis: On 12h timeframe, buy when price breaks above Camarilla H4 level with volume spike and 
daily price above weekly VWAP, sell when price breaks below L4 level with volume spike and daily 
price below weekly VWAP. Uses Camarilla pivot levels from daily chart and weekly VWAP for regime 
filter. Works in bull (breakouts above resistance) and bear (breakdowns below support) by fading 
to mean reversion in ranging markets (filtered by weekly VWAP). Target: 15-40 trades over 4 years 
(4-10/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H INDICATORS ===
    # Average True Range for volume spike detection
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.nanmean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Volume spike: volume > 1.5 * average volume (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.5)
    
    # === 1D INDICATOR: Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        range_ = high_1d[i] - low_1d[i]
        camarilla_h4[i] = close_1d[i] + range_ * 1.1 / 2
        camarilla_l4[i] = close_1d[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 1W INDICATOR: Weekly VWAP for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate VWAP for each week
    vwap_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        typical_price = (high_1w[i] + low_1w[i] + close_1w[i]) / 3
        vwap_1w[i] = typical_price  # Simplified: VWAP ≈ typical price for weekly
    
    # Align weekly VWAP to 12h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # start after warmup
        # Skip if any data not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vwap_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long setup: price breaks above H4 with volume spike and price above weekly VWAP
        long_setup = (high[i] > camarilla_h4_aligned[i] and 
                     volume_spike[i] and 
                     close[i] > vwap_1w_aligned[i])
        
        # Short setup: price breaks below L4 with volume spike and price below weekly VWAP
        short_setup = (low[i] < camarilla_l4_aligned[i] and 
                      volume_spike[i] and 
                      close[i] < vwap_1w_aligned[i])
        
        # Exit conditions: opposite breakout or loss of volume/spike
        exit_long = (low[i] < camarilla_l4_aligned[i] and volume_spike[i]) or \
                   (not volume_spike[i] and position == 1)
        exit_short = (high[i] > camarilla_h4_aligned[i] and volume_spike[i]) or \
                    (not volume_spike[i] and position == -1)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals