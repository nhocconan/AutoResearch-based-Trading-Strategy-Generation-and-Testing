#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Churning Filter for medium-term trend following
# KAMA adapts to market noise, providing smooth trend direction
# RSI(14) filters overbought/oversold conditions in trending markets
# Churning filter (price vs KAMA slope) avoids whipsaws in ranging markets
# Designed for 1d timeframe targeting 30-100 trades over 4 years (7-25/year)
# Works in bull/bear: captures sustained trends, avoids false signals in consolidation

name = "1d_KAMA_RSI_ChurnFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on 1d timeframe
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        abs_change = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(1, len(close)):
            if np.sum(abs_change[i-length+1:i+1]) > 0:
                er[i] = change[i] / np.sum(abs_change[i-length+1:i+1])
            else:
                er[i] = 0
        
        # Smoothing Constant
        sc = (er * (2/(slow+1) - 2/(fast+1)) + 2/(fast+1)) ** 2
        
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate churning filter: price vs KAMA slope
    kama_slope = np.gradient(kama_vals)
    churning = np.abs(close - kama_vals) / (np.abs(kama_slope) + 1e-10)
    churn_threshold = pd.Series(churning).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: >1.3x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_vals[i]) or np.isnan(rsi[i]) or np.isnan(churn_threshold[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above KAMA, RSI > 50, low churning (trending)
            if (close[i] > kama_vals[i] and rsi[i] > 50 and 
                churning[i] < churn_threshold[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA, RSI < 50, low churning (trending)
            elif (close[i] < kama_vals[i] and rsi[i] < 50 and 
                  churning[i] < churn_threshold[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI < 40
            if close[i] < kama_vals[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI > 60
            if close[i] > kama_vals[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals