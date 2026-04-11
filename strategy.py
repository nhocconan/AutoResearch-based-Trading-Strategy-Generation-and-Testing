#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla breakout with volume confirmation and ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts above/below these levels with volume confirmation and ADX > 25 indicate strong momentum. Works in bull markets via breakout continuations and bear markets via breakdown continuations. Designed for low trade frequency (~20-50/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h ADX(14) for trend strength
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14 + 1e-10))
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_14 + 1e-10))
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3/L3, H4/L4
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if np.isnan(adx[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_strong = adx[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h4_aligned[i] and close[i-1] <= camarilla_h4_aligned[i-1]
        breakdown_down = close[i] < camarilla_l4_aligned[i] and close[i-1] >= camarilla_l4_aligned[i-1]
        
        # Entry conditions
        # Long: Breakout above H4 AND volume confirmation AND strong trend
        if breakout_up and vol_confirm[i] and trend_strong and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Breakdown below L4 AND volume confirmation AND strong trend
        elif breakdown_down and vol_confirm[i] and trend_strong and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout or trend weakening
        elif position == 1 and (close[i] < camarilla_l3_aligned[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > camarilla_h3_aligned[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals