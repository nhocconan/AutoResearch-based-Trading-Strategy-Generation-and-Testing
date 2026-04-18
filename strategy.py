#!/usr/bin/env python3
"""
6h_Liquidity_Wave_Trend
Hypothesis: On 6-hour timeframe, identify directional momentum by combining 
liquidity-based mean reversion (price deviation from volume-weighted average price) 
with trend confirmation from higher timeframe (1-day EMA). 
In bull/bear markets, price tends to revert to VWAP during pullbacks in strong trends.
Uses VWAP deviation as entry signal with 1-day EMA as trend filter.
Designed for low trade frequency (target: 15-35/year) with controlled risk.
"""

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
    
    # Calculate 6-hour VWAP (volume-weighted average price)
    # VWAP = cumulative(volume * typical_price) / cumulative(volume)
    # where typical_price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(volume * typical_price)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Calculate price deviation from VWAP as percentage
    # Negative deviation = price below VWAP (potential long setup in uptrend)
    # Positive deviation = price above VWAP (potential short setup in downtrend)
    vwap_deviation = (close - vwap) / vwap * 100.0
    
    # Calculate 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 calculation
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha + ema50_1d[i-1] * (1 - alpha)
    
    # Align 1-day EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6-hour ATR for dynamic thresholds
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.nanmean(tr[1:15])  # First ATR as simple average
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Dynamic entry threshold: 0.5 * ATR as percentage of price
    # This adapts to volatility - wider bands in volatile markets
    dynamic_threshold = 0.5 * (atr / close) * 100.0  # Convert to percentage
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_deviation[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(dynamic_threshold[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price significantly below VWAP in uptrend (price > EMA50)
            if (vwap_deviation[i] < -dynamic_threshold[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price significantly above VWAP in downtrend (price < EMA50)
            elif (vwap_deviation[i] > dynamic_threshold[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or trend breaks down
            if (vwap_deviation[i] > -0.5 * dynamic_threshold[i] or  # Halfway back to VWAP
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or trend breaks up
            if (vwap_deviation[i] < 0.5 * dynamic_threshold[i] or  # Halfway back to VWAP
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Liquidity_Wave_Trend"
timeframe = "6h"
leverage = 1.0