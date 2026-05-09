#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Breakout_Volume_Trend_Filter"
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
    
    # Daily Camarilla pivot levels (using previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    
    range_hl = prev_high - prev_low
    camarilla_h4 = prev_close + 1.5 * range_hl
    camarilla_l4 = prev_close - 1.5 * range_hl
    camarilla_h3 = prev_close + 1.125 * range_hl
    camarilla_l3 = prev_close - 1.125 * range_hl
    camarilla_h2 = prev_close + 0.75 * range_hl
    camarilla_l2 = prev_close - 0.75 * range_hl
    camarilla_h1 = prev_close + 0.5 * range_hl
    camarilla_l1 = prev_close - 0.5 * range_hl
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_12h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_12h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_12h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_12h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Weekly trend filter: EMA50 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.8 * 24-period average (24*12h = 12 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > 1.8 * vol_ma24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or np.isnan(h3_12h[i]) or 
            np.isnan(l3_12h[i]) or np.isnan(h2_12h[i]) or np.isnan(l2_12h[i]) or
            np.isnan(h1_12h[i]) or np.isnan(l1_12h[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: Price breaks above H3 with volume confirmation and weekly uptrend
            if (price > h3_12h[i] and 
                vol_confirm[i] and 
                price > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short conditions: Price breaks below L3 with volume confirmation and weekly downtrend
            elif (price < l3_12h[i] and 
                  vol_confirm[i] and 
                  price < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: Price breaks below L3 (reversal) or weekly trend turns down
            if (price < l3_12h[i] or price < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above H3 (reversal) or weekly trend turns up
            if (price > h3_12h[i] or price > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals