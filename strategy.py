#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_MeanReversion
# Hypothesis: Combines KAMA trend direction with RSI mean-reversion for entries in both trending and ranging markets.
# Uses KAMA(10) for trend direction, RSI(14) for overbought/oversold levels, and volume confirmation.
# Designed to capture trend continuations in bull markets and mean-reversion bounces in bear markets.
# Target: 20-35 trades/year per symbol with disciplined risk control.

name = "4h_KAMA_Trend_RSI_MeanReversion"
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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # Calculate RSI
    def rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period-1] = np.mean(gain[0:period])
        avg_loss[period-1] = np.mean(loss[0:period])
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    # Get KAMA and RSI
    kama_vals = kama(close, 10, 2, 30)
    rsi_vals = rsi(close, 14)
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    else:
        vol_ma[:] = np.mean(volume) if len(volume) > 0 else 1
    
    volume_ratio = np.zeros_like(volume)
    valid_vol = vol_ma != 0
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction: price above/below KAMA
        trend_up = close[i] > kama_vals[i]
        trend_down = close[i] < kama_vals[i]
        
        if position == 0:
            # Enter long: Uptrend + RSI oversold + volume confirmation
            if trend_up and rsi_vals[i] < 30 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + RSI overbought + volume confirmation
            elif trend_down and rsi_vals[i] > 70 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend reversal or RSI overbought
            if not trend_up or rsi_vals[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend reversal or RSI oversold
            if not trend_down or rsi_vals[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals