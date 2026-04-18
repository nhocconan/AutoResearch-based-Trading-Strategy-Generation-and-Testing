#!/usr/bin/env python3
"""
12h_KAMA_RSI_Trend_Filter
Hypothesis: Use KAMA to identify trend direction, combined with RSI for momentum confirmation and volume filter for validation.
Works in both bull and bear markets by following the trend direction while avoiding whipsaws through RSI extremes and volume confirmation.
Target: 12-30 trades per year (48-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA trend indicator
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close.shape) > 1 else np.abs(np.diff(close)).sum()
        # Handle 1D case
        if len(close.shape) == 1:
            volatility = np.sum(np.abs(np.diff(close)))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10, 2, 30)
    
    # RSI(14) for momentum
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close, 14)
    
    # Volume filter: above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        price = close[i]
        kama_val = kama_vals[i]
        rsi_val = rsi_vals[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA, RSI not overbought, volume confirmation
            if price > kama_val and rsi_val < 70 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI not oversold, volume confirmation
            elif price < kama_val and rsi_val > 30 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below KAMA or RSI overbought
            if price < kama_val or rsi_val > 75:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above KAMA or RSI oversold
            if price > kama_val or rsi_val < 25:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_RSI_Trend_Filter"
timeframe = "12h"
leverage = 1.0