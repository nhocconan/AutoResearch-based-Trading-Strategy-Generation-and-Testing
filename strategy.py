#!/usr/bin/env python3
"""
6h_Adaptive_RSI_Trend_Volume
Hypothesis: On 6h timeframe, use adaptive RSI (based on volatility) combined with 1-day EMA trend filter and volume spike for entries.
In high volatility periods, RSI thresholds widen to capture trends; in low volatility, they narrow for mean reversion.
This adapts to both bull and bear markets by adjusting sensitivity to market conditions.
Target: 15-25 trades/year to minimize fee drag while maintaining edge.
"""

name = "6h_Adaptive_RSI_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get price, volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Adaptive RSI thresholds based on ATR ratio (current ATR vs 50-period average)
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_50 + 1e-10)
    
    # In high volatility (atr_ratio > 1.2): widen thresholds for trend following
    # In low volatility (atr_ratio < 0.8): narrow thresholds for mean reversion
    # Normal: use standard 30/70
    rsi_long_threshold = np.where(atr_ratio > 1.2, 20,
                     np.where(atr_ratio < 0.8, 40, 30))
    rsi_short_threshold = np.where(atr_ratio > 1.2, 80,
                      np.where(atr_ratio < 0.8, 60, 70))
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI(14), ATR(14,50), EMA34
    start_idx = max(14, 50, 34)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(rsi_long_threshold[i]) or
            np.isnan(rsi_short_threshold[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI below adaptive long threshold with uptrend and volume
            if rsi[i] < rsi_long_threshold[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI above adaptive short threshold with downtrend and volume
            elif rsi[i] > rsi_short_threshold[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI crosses above 50 (momentum fade) or trend change
            if rsi[i] > 50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI crosses below 50 (momentum fade) or trend change
            if rsi[i] < 50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals