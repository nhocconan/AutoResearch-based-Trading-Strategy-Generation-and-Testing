#!/usr/bin/env python3
"""
6h_Pivot_Reversal_Confluence
Hypothesis: Price reversals at key pivot levels (Camarilla R3/S3) with volume divergence and 1d trend alignment yield high-probability entries. Works in both bull and bear markets by fading extremes in ranging markets and catching reversals in trending markets. Target: 15-30 trades/year.
"""

name = "6h_Pivot_Reversal_Confluence"
timeframe = "6h"
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
    
    # Calculate Camarilla pivot levels from previous day
    # Using daily high/low/close from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    R3 = pivot + (range_hl * 1.1 / 2)
    S3 = pivot - (range_hl * 1.1 / 2)
    R4 = pivot + (range_hl * 1.1)
    S4 = pivot - (range_hl * 1.1)
    
    # Align daily levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d trend filter (EMA50)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_6h = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_6h = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume divergence: current volume < average volume (sign of exhaustion)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_exhaustion = volume < vol_ma  # Lower volume on test = exhaustion
    
    # RSI for overbought/oversold confirmation
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if pivot levels not available (first day)
        if np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        rsi_val = rsi[i]
        vol_exhaust = volume_exhaustion[i]
        uptrend = uptrend_1d_6h[i]
        downtrend = downtrend_1d_6h[i]
        
        if position == 0:
            # LONG: Price at S3 support with RSI oversold, volume exhaustion, and 1d uptrend bias
            if price <= S3_6h[i] and rsi_val < 30 and vol_exhaust and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R3 resistance with RSI overbought, volume exhaustion, and 1d downtrend bias
            elif price >= R3_6h[i] and rsi_val > 70 and vol_exhaust and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R3 or RSI overbought or trend changes
            if price >= R3_6h[i] or rsi_val > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S3 or RSI oversold or trend changes
            if price <= S3_6h[i] or rsi_val < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals