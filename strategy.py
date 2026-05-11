#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend signals in both trending and ranging markets.
Combined with RSI(14) > 50 for long and < 50 for short filters to avoid counter-trend entries.
Uses volume confirmation (volume > 1.3x 20-period average) to filter false signals.
Designed for 20-40 trades/year per symbol to avoid fee drag while capturing trends.
Works in both bull and bear markets by adapting to market conditions via KAMA's efficiency ratio.
"""

name = "4h_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- KAMA Calculation ---
    # Efficiency Ratio
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.sum(np.abs(np.diff(close_4h)), axis=0)  # placeholder, will compute properly below
    
    # Proper volatility calculation (sum of absolute changes over period)
    volatility_raw = np.abs(np.diff(close_4h))
    volatility = np.zeros_like(volatility_raw)
    for i in range(len(volatility_raw)):
        if i < 10:
            volatility[i] = np.sum(volatility_raw[:i+1]) if i > 0 else volatility_raw[0]
        else:
            volatility[i] = np.sum(volatility_raw[i-9:i+1])
    
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # --- RSI Calculation ---
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Average ---
    vol_avg = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                # Check stoploss (1.5x ATR from entry)
                atr_est = np.max([high_4h[i] - low_4h[i], 
                                 np.abs(high_4h[i] - close_4h[i-1]),
                                 np.abs(low_4h[i] - close_4h[i-1])]) if i > 0 else high_4h[i] - low_4h[i]
                if position == 1 and close_4h[i] <= entry_price - 1.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 1.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_4h[i] > 1.3 * vol_avg[i]
        
        if position == 0:
            # Look for entries
            if vol_confirm:
                # Long: price above KAMA AND RSI > 50
                if close_4h[i] > kama[i] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_4h[i]
                # Short: price below KAMA AND RSI < 50
                elif close_4h[i] < kama[i] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit when price crosses below KAMA OR RSI < 40
                if close_4h[i] < kama[i] or rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit when price crosses above KAMA OR RSI > 60
                if close_4h[i] > kama[i] or rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals