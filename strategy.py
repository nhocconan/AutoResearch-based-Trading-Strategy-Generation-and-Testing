#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_breakout_v1
# Combines 12h Camarilla pivot levels for primary direction and 1d volume confirmation
# with 12h ADX trend filter. Designed to capture institutional breakouts in trending
# markets while avoiding false signals in ranging conditions. Target: 25-40 trades/year.
name = "4h_12h_camarilla_breakout_v1"
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
    
    # Get 12h data for Camarilla and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels from previous day
    high_prev = df_12h['high'].shift(1).values
    low_prev = df_12h['low'].shift(1).values
    close_prev = df_12h['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align 12h levels to 4h
    h3_level = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    h4_level = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # 12h ADX for trend filter (ADX > 25 = trending)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    trend_filter = adx_aligned > 25
    
    # 1d volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check filters
        if not (trend_filter[i] and vol_confirm[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H3 with volume and trend
        if close[i] > h3_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L3 with volume and trend
        elif close[i] < l3_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite H4/L4 breakout
        elif close[i] > h4_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] < l4_level[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals