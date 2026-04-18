#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Confirmation_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction and RSI for momentum confirmation on 4h timeframe.
Go long when price is above KAMA and RSI > 55, short when price is below KAMA and RSI < 45.
Requires volume > 1.5x 20-period average for confirmation.
Target: 20-30 trades/year by using trend-following with momentum filter to reduce noise.
Works in bull markets via trend following and in bear via short signals.
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
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close, kama_period))
    abs_diff = np.abs(np.diff(close))
    er_num = change[kama_period-1:]  # length n - kama_period + 1
    er_den = np.array([np.sum(abs_diff[i:i+kama_period]) for i in range(len(abs_diff)-kama_period+1)])
    er = np.divide(er_num, er_den, out=np.full_like(er_num, 0.0), where=er_den!=0)
    
    # Calculate smoothing constant SC
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[kama_period-1] = close[kama_period-1]  # seed
    
    for i in range(kama_period, len(close)):
        kama[i] = kama[i-1] + sc[i-kama_period] * (close[i] - kama[i-1])
    
    # RSI(14) calculation
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 0.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full_like(volume, np.nan)
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kama_period, rsi_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA and RSI > 55 and volume confirmation
            if close[i] > kama[i] and rsi[i] > 55 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI < 45 and volume confirmation
            elif close[i] < kama[i] and rsi[i] < 45 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA OR RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA OR RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_RSI_Confirmation_v1"
timeframe = "4h"
leverage = 1.0