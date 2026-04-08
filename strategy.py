#!/usr/bin/env python3
"""
6H Triple Timeframe Confluence: 1h Trend + 12h Momentum + 6h Entry
Hypothesis: Combining 1h EMA trend filter, 12h RSI momentum, and 6h price action
creates high-probability entries with low frequency. The multi-timeframe alignment
reduces false signals while capturing trends in both bull and bear markets.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_triple_timeframe_confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    ema_20_1h = df_1h['close'].ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    # 12h data for momentum filter
    df_12h = get_htf_data(prices, '12h')
    rsi_14_12h = _calculate_rsi(df_12h['close'].values, 14)
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    # 6h data for entry signals
    atr_14_6h = _calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1h_aligned[i]) or 
            np.isnan(rsi_14_12h_aligned[i]) or 
            np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend breaks or momentum fades
            if close[i] < ema_20_1h_aligned[i] or rsi_14_12h_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend breaks or momentum fades
            if close[i] > ema_20_1h_aligned[i] or rsi_14_12h_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: uptrend + bullish momentum + pullback to support
            if (close[i] > ema_20_1h_aligned[i] and 
                rsi_14_12h_aligned[i] > 55 and 
                close[i] <= low[i-1] + atr_14_6h[i] * 0.5):
                position = 1
                signals[i] = 0.25
            # Short: downtrend + bearish momentum + bounce to resistance
            elif (close[i] < ema_20_1h_aligned[i] and 
                  rsi_14_12h_aligned[i] < 45 and 
                  close[i] >= high[i-1] - atr_14_6h[i] * 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals

def _calculate_rsi(prices, period):
    """Calculate RSI with proper handling of edge cases"""
    if len(prices) < period:
        return np.full_like(prices, np.nan, dtype=float)
    
    delta = np.diff(prices)
    delta = np.prepend(delta, np.nan)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    
    # First average
    avg_gain[period] = np.nanmean(gain[1:period+1])
    avg_loss[period] = np.nanmean(loss[1:period+1])
    
    # Subsequent averages
    for i in range(period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _calculate_atr(high, low, close, period):
    """Calculate ATR with proper handling"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    tr1 = high - low
    tr2 = np.abs(np.subtract(high, np.roll(close, 1)))
    tr3 = np.abs(np.subtract(low, np.roll(close, 1)))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's ATR
    atr = np.full_like(tr, np.nan, dtype=float)
    atr[period] = np.nanmean(tr[1:period+1])
    
    for i in range(period+1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr