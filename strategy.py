#!/usr/bin/env python3
"""
#100896 - 12h_KAMA_Direction_With_RSI_Chop_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) direction filter on 12h with RSI(14) and 
Choppiness Index regime filter to avoid whipsaw. KAMA adapts to market noise, reducing false signals. 
RSI avoids extremes, Choppiness Index identifies ranging vs trending markets. 
Targets 12-37 trades/year (50-150 total) by requiring confluence of trend, momentum, and regime.
Works in bull (KAMA up + RSI>50 + trending) and bear (KAMA down + RSI<50 + trending).
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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.abs(np.diff(close_prices))
        er = np.zeros_like(close_prices)
        for i in range(length, len(close_prices)):
            if volatility[i-length+1:i+1].sum() != 0:
                er[i] = change[i-length+1:i+1].sum() / volatility[i-length+1:i+1].sum()
            else:
                er[i] = 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index(14)
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=length, min_periods=length).sum().values
        
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if atr[i] != 0 and (highest_high[i] - lowest_low[i]) != 0:
                chop[i] = 100 * np.log10(atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    chop = choppiness_index(high, low, close, length=14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: KAMA up, RSI > 50, trending market (CHOP < 38.2), volume spike
        if (close[i] > kama[i] and 
            rsi[i] > 50 and 
            chop[i] < 38.2 and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: KAMA down, RSI < 50, trending market (CHOP < 38.2), volume spike
        elif (close[i] < kama[i] and 
              rsi[i] < 50 and 
              chop[i] < 38.2 and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: reverse signal or choppy market
        elif position == 1 and (close[i] < kama[i] or chop[i] > 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > kama[i] or chop[i] > 61.8):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Direction_With_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0