#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, L3, H4, L4
    # H3 = close + 1.1*(high - low)/6
    # L3 = close - 1.1*(high - low)/6
    # H4 = close + 1.1*(high - low)/2
    # L4 = close - 1.1*(high - low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: 4h volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough data for 12h EMA50 and vol MA20
    
    for i in range(start_idx, n):
        # Skip if 12h trend data not ready
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma20[i] if not np.isnan(vol_ma20[i]) else False
        
        if position == 0:
            # Long: price breaks above Camarilla H3 + 12h uptrend + volume confirmation
            if (close[i] > camarilla_h3_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 + 12h downtrend + volume confirmation
            elif (close[i] < camarilla_l3_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla L3 or 12h trend turns down
            if (close[i] < camarilla_l3_aligned[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla H3 or 12h trend turns up
            if (close[i] > camarilla_h3_aligned[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals