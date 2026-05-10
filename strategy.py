#!/usr/bin/env python3
# 4h_KAMA_Trend_Reversal_With_Volume_Confirmation
# Hypothesis: Long when KAMA turns bullish (close > KAMA) with volume > 1.3x average in uptrend (price > 1d EMA50).
# Short when KAMA turns bearish (close < KAMA) with volume > 1.3x average in downtrend (price < 1d EMA50).
# Exit when price crosses KAMA in opposite direction or ATR-based stoploss hit.
# Uses Kaufman Adaptive Moving Average for trend following, works in both bull and bear markets.
# Designed for 20-50 trades/year to avoid fee drag.

name = "4h_KAMA_Trend_Reversal_With_Volume_Confirmation"
timeframe = "4h"
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Kaufman Adaptive Moving Average (KAMA)
    # Parameters: ER period=10, fast=2, slow=30
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Pad volatility to match change length
    volatility = np.concatenate([np.full(er_period-1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constant
    sc = np.power(er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1), 2)
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # Initialize
    for i in range(er_period+1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period+1, 20, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1d EMA50 trend
            if close[i] > ema_50_1d_aligned[i]:  # Uptrend
                # Long: Price crosses above KAMA with volume confirmation
                if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume[i] > 1.3 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Price crosses below KAMA with volume confirmation
                if close[i] < kama[i] and close[i-1] >= kama[i-1] and volume[i] > 1.3 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price crosses below KAMA or stoploss hit
            if close[i] < kama[i] or (i > 0 and low[i] < kama[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above KAMA or stoploss hit
            if close[i] > kama[i] or (i > 0 and high[i] > kama[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals