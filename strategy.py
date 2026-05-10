#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_1wVWAP_Filter
Hypothesis: On daily chart, KAMA trend direction combined with price above/below weekly VWAP 
provides robust trend following that works in both bull and bear markets. Weekly VWAP acts 
as dynamic support/resistance, reducing false signals. Low turnover expected (<15 trades/year).
"""

name = "1d_KAMA_Trend_With_1wVWAP_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, 0)  # align length
    
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    er = np.zeros(n)
    for i in range(1, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_num = (typical_price_1w * df_1w['volume']).values
    vwap_den = df_1w['volume'].values
    
    vwap_cumsum_num = np.cumsum(vwap_num)
    vwap_cumsum_den = np.cumsum(vwap_den)
    vwap_1w = np.divide(vwap_cumsum_num, vwap_cumsum_den, 
                        out=np.full_like(vwap_cumsum_num, np.nan), 
                        where=vwap_cumsum_den!=0)
    
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # KAMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(vwap_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters
        price_above_kama = close[i] > kama[i]
        price_above_vwap = close[i] > vwap_1w_aligned[i]
        
        if position == 0:
            # Long: Price above both KAMA and weekly VWAP
            if price_above_kama and price_above_vwap:
                signals[i] = 0.25
                position = 1
            # Short: Price below both KAMA and weekly VWAP
            elif not price_above_kama and not price_above_vwap:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below KAMA OR below VWAP
            if not (price_above_kama and price_above_vwap):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above KAMA OR above VWAP
            if price_above_kama or price_above_vwap:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals