#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted RSI with 12h Supertrend filter and ATR-based stops.
- VW-RSI(14): RSI calculated using typical price * volume as input, reducing noise from low-volume spikes
- Long when VW-RSI crosses above 30 AND 12h Supertrend is bullish
- Short when VW-RSI crosses below 70 AND 12h Supertrend is bearish
- Exit when VW-RSI returns to 50 (mean reversion) OR Supertrend flips
- Uses 6h primary with 12h HTF for trend filter to avoid counter-trend trades
- Volume weighting makes RSI more responsive to institutional participation
- Designed to work in ranging markets (mean reversion at extremes) and trending markets (pullbacks in trend)
- Signal size: 0.25 discrete levels
- Target: 80-180 total trades over 4 years (20-45/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Typical price for volume weighting
    typical_price = (high + low + close) / 3.0
    vol_typical = typical_price * volume
    
    # VW-RSI calculation (RSI on volume-weighted typical price)
    def rsi(values, period=14):
        delta = np.diff(values, prepend=values[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(values)
        avg_loss = np.zeros_like(values)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(values)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    vw_rsi = rsi(vol_typical, 14)
    
    # 12h Supertrend for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ATR calculation
    def atr(high, low, close, period=10):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr_vals = np.zeros_like(tr)
        atr_vals[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_12h = atr(high_12h, low_12h, close_12h, 10)
    
    # Supertrend calculation
    def supertrend(high, low, close, atr_vals, period=10, multiplier=3.0):
        hl2 = (high + low) / 2.0
        upperband = hl2 + (multiplier * atr_vals)
        lowerband = hl2 - (multiplier * atr_vals)
        
        # Initialize
        final_upperband = np.zeros_like(close)
        final_lowerband = np.zeros_like(close)
        supertrend_vals = np.zeros_like(close)
        trend = np.ones_like(close, dtype=int)  # 1 for uptrend, -1 for downtrend
        
        final_upperband[0] = upperband[0]
        final_lowerband[0] = lowerband[0]
        supertrend_vals[0] = final_lowerband[0]
        trend[0] = 1
        
        for i in range(1, len(close)):
            # Upper band logic
            if upperband[i] < final_upperband[i-1] or close[i-1] > final_upperband[i-1]:
                final_upperband[i] = upperband[i]
            else:
                final_upperband[i] = final_upperband[i-1]
            
            # Lower band logic
            if lowerband[i] > final_lowerband[i-1] or close[i-1] < final_lowerband[i-1]:
                final_lowerband[i] = lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]
            
            # Trend logic
            if trend[i-1] == -1 and close[i] > final_upperband[i]:
                trend[i] = 1
            elif trend[i-1] == 1 and close[i] < final_lowerband[i]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]
            
            # Supertrend value
            if trend[i] == 1:
                supertrend_vals[i] = final_lowerband[i]
            else:
                supertrend_vals[i] = final_upperband[i]
        
        return supertrend_vals, trend
    
    supertrend_12h, trend_12h = supertrend(high_12h, low_12h, close_12h, atr_12h, 10, 3.0)
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend_12h)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 30) + 5  # Need VW-RSI, Supertrend warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vw_rsi[i]) or np.isnan(supertrend_12h_aligned[i]) or 
            np.isnan(trend_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: VW-RSI crosses above 30 AND 12h Supertrend bullish
            if vw_rsi[i] > 30 and vw_rsi[i-1] <= 30 and trend_12h_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: VW-RSI crosses below 70 AND 12h Supertrend bearish
            elif vw_rsi[i] < 70 and vw_rsi[i-1] >= 70 and trend_12h_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VW-RSI returns to 50 OR Supertrend turns bearish
            if vw_rsi[i] >= 50 or trend_12h_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: VW-RSI returns to 50 OR Supertrend turns bullish
            if vw_rsi[i] <= 50 or trend_12h_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolWeightedRSI_12hSupertrend_v1"
timeframe = "6h"
leverage = 1.0