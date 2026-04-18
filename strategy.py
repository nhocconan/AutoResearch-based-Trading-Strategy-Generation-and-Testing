#!/usr/bin/env python3
"""
1d_1w_KAMA_Direction_RSI_Pullback_V1
Hypothesis: Trade pullbacks in direction of weekly KAMA trend on daily chart using RSI(14) oversold/overbought levels. Uses weekly trend filter to avoid counter-trend trades. Position size 0.25 targeting ~20 trades/year to minimize fee decay. Works in bull/bear by trading with trend and mean-reversion on pullbacks.
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
    
    # Get daily data for RSI and price action
    df_1d = get_htf_data(prices, '1d')
    
    # Get weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    rsi = np.full_like(close, np.nan)
    
    rsi_period = 14
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
            rs = np.where(avg_loss[i] != 0, avg_gain[i] / avg_loss[i], 100)
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Weekly KAMA(30, 2, 30) - ER based adaptive moving average
    close_1w = df_1w['close'].values
    kama = np.full_like(close_1w, np.nan)
    
    kama_period = 30
    fast_sc = 2
    slow_sc = 30
    
    if len(close_1w) >= kama_period:
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close_1w, kama_period))
        volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
        er = np.zeros_like(change)
        for i in range(len(change)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        kama[kama_period-1] = np.mean(close_1w[:kama_period])
        for i in range(kama_period, len(close_1w)):
            kama[i] = kama[i-1] + sc[i - kama_period + 1] * (close_1w[i] - kama[i-1])
    
    # Align daily RSI to daily timeframe (already aligned)
    # Align weekly KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, rsi_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(rsi[i]) or np.isnan(kama_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly KAMA and RSI oversold (<30)
            if close[i] > kama_aligned[i] and rsi[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA and RSI overbought (>70)
            elif close[i] < kama_aligned[i] and rsi[i] > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or price crosses below weekly KAMA
            if rsi[i] > 70 or close[i] < kama_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or price crosses above weekly KAMA
            if rsi[i] < 30 or close[i] > kama_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Direction_RSI_Pullback_V1"
timeframe = "1d"
leverage = 1.0