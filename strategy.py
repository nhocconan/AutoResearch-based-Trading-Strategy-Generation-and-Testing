#!/usr/bin/env python3
# 6h_VWAP_Cross_RSI_Filter
# Hypothesis: VWAP cross combined with RSI momentum filter on 6h timeframe. Long when price crosses above VWAP and RSI > 50, short when price crosses below VWAP and RSI < 50. VWAP provides institutional reference, RSI filters momentum direction. Designed for 6h to achieve 12-37 trades/year, works in both bull and bear markets by following intraday momentum.

name = "6h_VWAP_Cross_RSI_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # RSI calculation (14-period)
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        # First average
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        # Pad first element
        rsi_full = np.full_like(prices, np.nan)
        rsi_full[1:] = rsi
        return rsi_full
    
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough history for RSI
    
    for i in range(start_idx, n):
        if np.isnan(vwap[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP and RSI > 50
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP and RSI < 50
            elif close[i] < vwap[i] and close[i-1] >= vwap[i-1] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below VWAP
            if close[i] < vwap[i] and close[i-1] >= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above VWAP
            if close[i] > vwap[i] and close[i-1] <= vwap[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals