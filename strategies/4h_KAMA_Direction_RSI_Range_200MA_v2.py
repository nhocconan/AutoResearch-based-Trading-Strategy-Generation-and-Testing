#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Range_200MA_v2
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI for momentum confirmation and 200-period MA for long-term bias.
Designed to capture strong trends while filtering choppy markets, with low trade frequency
to minimize fee drag. Targets 20-50 trades per year for robustness in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive trend) - faster reaction in trends, slower in chop
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper KAMA calculation
    diff = np.abs(np.diff(close, prepend=close[0]))
    # Volatility sum over 10 periods
    volatility_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_sum[i] = np.sum(diff[i-9:i+1])
    er = np.where(volatility_sum > 0, diff / volatility_sum, 0)
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2
    
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 200-period MA for long-term trend filter
    ma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ma200[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        ma200_val = ma200[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long conditions: price above KAMA (uptrend), RSI > 50 (momentum), above MA200 (long-term bias), volume confirmation
            if close_val > kama_val and rsi_val > 50 and close_val > ma200_val and vol_conf:
                signals[i] = size
                position = 1
            # Short conditions: price below KAMA (downtrend), RSI < 50, below MA200, volume confirmation
            elif close_val < kama_val and rsi_val < 50 and close_val < ma200_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI < 40 (loss of momentum)
            if close_val < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI > 60 (loss of momentum)
            if close_val > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Direction_RSI_Range_200MA_v2"
timeframe = "4h"
leverage = 1.0