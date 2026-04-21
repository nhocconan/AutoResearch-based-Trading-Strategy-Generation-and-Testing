#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2
Hypothesis: On 1d timeframe, price breaking above Camarilla R1 or below S1 with volume confirmation 
and ATR filter provides high-probability continuation trades. Weekly trend filter (price vs weekly EMA20) 
ensures alignment with higher timeframe trend, reducing whipsaw in counter-trend markets. 
Works in bull/bear by taking both long and short breaks with proper filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    multiplier = 1.1 / 12
    close_val = close
    R4 = close_val + range_val * 1.1 * 0.5
    R3 = close_val + range_val * 1.1 * 0.25
    R2 = close_val + range_val * 1.1 * 0.1666
    R1 = close_val + range_val * 1.1 * 0.0833
    S1 = close_val - range_val * 1.1 * 0.0833
    S2 = close_val - range_val * 1.1 * 0.1666
    S3 = close_val - range_val * 1.1 * 0.25
    S4 = close_val - range_val * 1.1 * 0.5
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1_1d = np.full_like(close_1d, np.nan)
    S1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            R1_1d[i] = np.nan
            S1_1d[i] = np.nan
        else:
            R1, _, _, _, S1, _, _, _ = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
            R1_1d[i] = R1
            S1_1d[i] = S1
    
    # Align Camarilla levels to 1d timeframe (they're already daily, but align for safety)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2/21) + (ema_20_1w[i-1] * 19/21)
    
    # Align weekly EMA20 to 1d timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume and ATR on 1d
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    atr_14 = np.full_like(close_1d, np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Align volume and ATR to 1d timeframe
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume average for confirmation
    vol_avg_20 = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-20:i])
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if NaN in critical values
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]  # use 1d close for signal generation
        R1 = R1_1d_aligned[i]
        S1 = S1_1d_aligned[i]
        ema20_w = ema_20_1w_aligned[i]
        volume = volume_1d_aligned[i]
        vol_avg = vol_avg_20_aligned[i]
        atr = atr_14_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume > vol_avg * 1.5
        
        # ATR filter: only trade if ATR is reasonable (avoid extremely low volatility)
        atr_filter = atr > 0 and atr < price * 0.1  # ATR less than 10% of price
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + ATR filter + above weekly EMA20
            if price > R1 and vol_confirm and atr_filter and price > ema20_w:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + ATR filter + below weekly EMA20
            elif price < S1 and vol_confirm and atr_filter and price < ema20_w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below S1 or weekly trend turns bearish
            if price < S1 or price < ema20_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or weekly trend turns bullish
            if price > R1 or price > ema20_w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v2"
timeframe = "1d"
leverage = 1.0