#!/usr/bin/env python3
"""
6h_RSI2_Regime_Breakout
6h strategy combining 2-period RSI mean reversion with daily regime filter.
- Long: RSI(2) < 10 + price > daily EMA50 (bull regime)
- Short: RSI(2) > 90 + price < daily EMA50 (bear regime)
- Exit: RSI(2) crosses back to neutral (40-60 range) or opposite signal
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (buy pullbacks) and bear markets (sell rallies)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for regime filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(2) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2  # 2-period smoothing
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need 50 for EMA50 + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: daily EMA50
        bull_regime = close[i] > ema_50_1d_aligned[i]
        bear_regime = close[i] < ema_50_1d_aligned[i]
        
        # RSI(2) signals
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        if position == 0:
            # Long: bull regime + RSI oversold
            if bull_regime and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short: bear regime + RSI overbought
            elif bear_regime and rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or bear regime
            if rsi_neutral[i] or not bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or bull regime
            if rsi_neutral[i] or not bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI2_Regime_Breakout"
timeframe = "6h"
leverage = 1.0