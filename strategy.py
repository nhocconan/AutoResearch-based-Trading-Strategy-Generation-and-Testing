#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA Trend + Volume Spike + RSI Momentum
# - Uses KAMA (Kaufman Adaptive Moving Average) from 4h to capture adaptive trend
# - RSI(14) for momentum confirmation (avoid overbought/oversold extremes)
# - Volume spike (>2x 20-period average) confirms breakout strength
# - Works in bull/bear by using KAMA direction as trend filter
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_KAMA_Volume_RSI"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) calculation
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for KAMA and RSI to stabilize
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI > 50 (momentum) + volume spike
            long_cond = (close[i] > kama[i] and 
                        rsi[i] > 50 and
                        volume_spike[i])
            
            # Short: price below KAMA (downtrend) + RSI < 50 (momentum) + volume spike
            short_cond = (close[i] < kama[i] and 
                         rsi[i] < 50 and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA (trend change) OR RSI < 40 (losing momentum)
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA (trend change) OR RSI > 60 (losing momentum)
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals