#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict
Hypothesis: Price breaking above Camarilla R1 or below S1 with volume confirmation and tight range filter captures institutional breakouts in both bull and bear markets. The range filter (low ATR percentile) ensures we only trade during low volatility, reducing false breakouts. Target: 20-30 trades/year (80-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    range_val = prev_high - prev_low
    camarilla_r1 = prev_close + range_val * 1.1 / 12
    camarilla_s1 = prev_close - range_val * 1.1 / 12
    
    # Align to 4h timeframe
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Range filter: ATR(14) < 30th percentile of ATR(50) -> low volatility regime
    tr1 = high[1:] - low[:-1]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum.reduce([tr1, tr2, tr3])])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    # Calculate percentile rank of ATR(14) over 50-period window
    atr_percentile = pd.Series(atr_14).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    range_filter = atr_percentile < 0.3  # Low volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for ATR percentile
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_4h[i]) or np.isnan(camarilla_s1_4h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(range_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_r1_4h[i]
        s1 = camarilla_s1_4h[i]
        vol_ok = volume_filter[i]
        range_ok = range_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in low volatility
            if price > r1 and vol_ok and range_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in low volatility
            elif price < s1 and vol_ok and range_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price crosses below S1 or volatility increases
            if price < s1 or not range_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price crosses above R1 or volatility increases
            if price > r1 or not range_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict"
timeframe = "4h"
leverage = 1.0