#!/usr/bin/env python3
"""
4h_KAMA_Direction_Volume_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Breakouts above/below KAMA bands with volume confirmation and daily EMA trend filter capture
strong institutional moves while filtering noise. Target: 20-40 trades/year (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    er = change / (volatility + 1e-10)
    er = np.concatenate([np.zeros(er_length-1), er])
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate ATR for bands
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close)
    atr[0] = np.mean(tr[:14]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr = np.concatenate([[atr[0]], atr])
    
    # KAMA bands
    upper_band = kama + 1.0 * atr
    lower_band = kama - 1.0 * atr
    
    # 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_1d_4h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_1d_4h[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price crosses above upper band with volume in uptrend
            if price > upper_band[i] and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below lower band with volume in downtrend
            elif price < lower_band[i] and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to KAMA or trend reverses
            if price < kama[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to KAMA or trend reverses
            if price > kama[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0