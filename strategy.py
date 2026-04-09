#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: 1d strategy using KAMA direction filter + RSI extremes + Choppiness regime for mean reversion in ranging markets and trend following in trending markets.
# Enters long when price is below KAMA, RSI < 30, and CHOP > 61.8 (ranging market mean reversion).
# Enters short when price is above KAMA, RSI > 70, and CHOP > 61.8 (ranging market mean reversion).
# Uses weekly trend filter: only take longs when price above weekly KAMA, shorts when price below weekly KAMA.
# Uses discrete position sizing (±0.25) to minimize fee churn.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull/bear via regime adaptation and weekly trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
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
    
    # Get 1d data for KAMA, RSI, and Choppiness calculation
    # KAMA calculation
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # Handle array case properly
        if len(close) > length:
            change = np.abs(np.diff(close, n=length))
            volatility = []
            for i in range(length, len(close)):
                volatility.append(np.sum(np.abs(np.diff(close[i-length:i+1]))))
            volatility = np.array(volatility)
            er = np.zeros(len(close))
            er[length:] = change / (volatility + 1e-10)
            er = np.where(np.isnan(er), 0, er)
            er = np.where(er > 1, 1, er)
        else:
            er = np.zeros(len(close))
        
        # Smoothing constants
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama = np.zeros(len(close))
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        
        # Wilder's smoothing
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi = np.concatenate([np.full(length, 50), rsi[length:]])  # Pad beginning
        return rsi
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, length=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], tr])
        
        # Sum of TR over period
        tr_sum = pd.Series(tr).rolling(window=length, min_periods=length).sum().values
        
        # Highest high and lowest low over period
        hh = pd.Series(high).rolling(window=length, min_periods=length).max().values
        ll = pd.Series(low).rolling(window=length, min_periods=length).min().values
        
        # Choppiness Index
        chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(length)
        chop = np.where((hh - ll) == 0, 50, chop)  # Avoid division by zero
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, length=14)
    chop = calculate_chop(high, low, close, length=14)
    
    # Get 1w HTF data ONCE before loop for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, length=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above KAMA OR RSI > 50 (mean reversion complete)
            if close[i] > kama[i] or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below KAMA OR RSI < 50 (mean reversion complete)
            if close[i] < kama[i] or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price below KAMA, RSI oversold, choppy market, and above weekly KAMA (uptrend filter)
            if (close[i] < kama[i]) and (rsi[i] < 30) and (chop[i] > 61.8) and (close[i] > kama_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price above KAMA, RSI overbought, choppy market, and below weekly KAMA (downtrend filter)
            elif (close[i] > kama[i]) and (rsi[i] > 70) and (chop[i] > 61.8) and (close[i] < kama_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals