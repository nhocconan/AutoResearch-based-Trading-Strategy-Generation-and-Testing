#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
in ranging markets it stays flat. Combined with RSI(14) for momentum confirmation
and volume filter to avoid false signals. Designed for 12h timeframe to target
50-150 total trades over 4 years (12-37/year) with low frequency to minimize fee drag.
Works in bull/bear via adaptive trend following and momentum confirmation.
"""

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
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
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14) for momentum confirmation
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        delta = np.concatenate([np.array([np.nan]), delta])
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        
        # First average
        if len(gain) > period:
            avg_gain[period] = np.nanmean(gain[1:period+1])
            avg_loss[period] = np.nanmean(loss[1:period+1])
            
            for i in range(period+1, len(gain)):
                if not np.isnan(avg_gain[i-1]):
                    avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                else:
                    avg_gain[i] = np.nan
                    
                if not np.isnan(avg_loss[i-1]):
                    avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
                else:
                    avg_loss[i] = np.nan
        
        # Avoid division by zero
        rs = np.full_like(avg_gain, np.nan)
        mask = avg_loss != 0
        rs[mask] = avg_gain[mask] / avg_loss[mask]
        
        rsi = np.full_like(avg_gain, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend: price above KAMA = uptrend, below = downtrend
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI levels: avoid overbought/oversold extremes
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI not overbought + volume
            if (price_above_kama and 
                rsi_not_overbought and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI not oversold + volume
            elif (price_below_kama and 
                  rsi_not_oversold and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or RSI overbought
            if (not price_above_kama) or (rsi[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or RSI oversold
            if (not price_below_kama) or (rsi[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals