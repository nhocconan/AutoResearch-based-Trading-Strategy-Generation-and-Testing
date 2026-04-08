#!/usr/bin/env python3
"""
12h_1d_camarilla_pivot_v1
Hypothesis: Trade reversals at 1-day Camarilla pivot levels with volume confirmation.
- Long at S1/S2 support levels when price bounces with volume spike
- Short at R1/R2 resistance levels when price rejects with volume spike
- Use 1-day ADX > 20 to filter choppy markets (avoid false signals in low volatility)
- Designed for low trade frequency (15-25/year) to minimize fee drag
- Works in bull/bear via mean reversion at key levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_v1"
timeframe = "12h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using Wilder's smoothing"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothing
    atr = np.zeros(len(high))
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros(len(high))
    minus_di = np.zeros(len(high))
    for i in range(period, len(high)):
        if atr[i] > 0:
            plus_di[i] = (np.sum(plus_dm[i-period+1:i+1]) / atr[i]) * 100
            minus_di[i] = (np.sum(minus_dm[i-period+1:i+1]) / atr[i]) * 100
    
    dx = np.zeros(len(high))
    for i in range(period, len(high)):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
    
    adx = np.zeros(len(high))
    adx[2*period-2] = np.mean(dx[period-1:2*period-1])
    for i in range(2*period-1, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h2 = np.zeros(len(close_1d))
    camarilla_l2 = np.zeros(len(close_1d))
    camarilla_h1 = np.zeros(len(close_1d))
    camarilla_l1 = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        if i == 0:
            # Use previous day's values (for first day, use same)
            camarilla_h4[i] = camarilla_l4[i] = camarilla_h3[i] = camarilla_l3[i] = camarilla_h2[i] = camarilla_l2[i] = camarilla_h1[i] = camarilla_l1[i] = close_1d[i]
        else:
            # Camarilla formulas using previous day's OHLC
            range_ = high_1d[i-1] - low_1d[i-1]
            close_prev = close_1d[i-1]
            
            camarilla_h4[i] = close_prev + range_ * 1.1 / 2
            camarilla_l4[i] = close_prev - range_ * 1.1 / 2
            camarilla_h3[i] = close_prev + range_ * 1.1 / 4
            camarilla_l3[i] = close_prev - range_ * 1.1 / 4
            camarilla_h2[i] = close_prev + range_ * 1.1 / 6
            camarilla_l2[i] = close_prev - range_ * 1.1 / 6
            camarilla_h1[i] = close_prev + range_ * 1.1 / 12
            camarilla_l1[i] = close_prev - range_ * 1.1 / 12
    
    # Calculate 1d ADX (14-period)
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        adx_val = adx_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price reaches H3 or volume drops
            if price >= camarilla_h3_aligned[i] or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price reaches L3 or volume drops
            if price <= camarilla_l3_aligned[i] or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches L4 with volume spike and trending market (ADX > 20)
            if price <= camarilla_l4_aligned[i] and vol_ratio > 1.8 and adx_val > 20:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H4 with volume spike and trending market (ADX > 20)
            elif price >= camarilla_h4_aligned[i] and vol_ratio > 1.8 and adx_val > 20:
                position = -1
                signals[i] = -0.25
    
    return signals