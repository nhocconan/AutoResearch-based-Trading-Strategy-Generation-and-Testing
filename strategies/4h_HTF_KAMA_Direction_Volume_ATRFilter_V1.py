#!/usr/bin/env python3
"""
4h_HTF_KAMA_Direction_Volume_ATRFilter_V1
Hypothesis: Use 1d KAMA direction as trend filter, enter on 4h break of prior 4-bar high/low with volume confirmation (>1.5x 20-bar MA), exit on ATR stoploss (2.0x) or opposite signal. KAMA adapts to market noise, reducing whipsaw in sideways markets. Volume confirmation ensures breakout legitimacy. Target 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for KAMA trend filter
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d KAMA for Trend Filter ===
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility properly: sum of abs changes over ER period
    er_period = 10
    volatility_sum = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close_1d[i-er_period:i+1])))
    # Avoid division by zero
    volatility_sum[volatility_sum == 0] = 1e-10
    er = np.zeros_like(close_1d)
    er[er_period:] = change[er_period:] / volatility_sum[er_period:]
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 4h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume MA (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        trend_up = kama_aligned[i] < price  # price above 1d KAMA = uptrend
        trend_down = kama_aligned[i] > price  # price below 1d KAMA = downtrend
        
        # Entry conditions: break of prior 4-bar high/low
        if i >= 4:
            prior_high = np.max(high[i-4:i])
            prior_low = np.min(low[i-4:i])
        else:
            prior_high = high[i]
            prior_low = low[i]
        
        if position == 0:
            # Long: break above prior 4-bar high with volume and uptrend
            if price > prior_high and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below prior 4-bar low with volume and downtrend
            elif price < prior_low and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < close[i-1] - 2.0 * atr[i] or (price < prior_low and vol_ok and trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > close[i-1] + 2.0 * atr[i] or (price > prior_high and vol_ok and trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_KAMA_Direction_Volume_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0