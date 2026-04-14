#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index + Daily KAMA reversal strategy
# In choppy markets (CHOP > 61.8), price tends to revert to the KAMA trend.
# Uses daily KAMA as dynamic trend filter and 12h Choppiness Index as regime filter.
# Works in both bull/bear markets by adapting to ranging conditions.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily KAMA (using ER=10, fast=2, slow=30)
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series.diff(10)).values
    volatility = abs(close_1d_series.diff(1)).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 12h Choppiness Index (14-period)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = np.zeros_like(close)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(close)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    max_high = np.zeros_like(high)
    min_low = np.zeros_like(low)
    max_high[atr_period-1] = np.max(high[:atr_period])
    min_low[atr_period-1] = np.min(low[:atr_period])
    for i in range(atr_period, len(close)):
        max_high[i] = max(max_high[i-1], high[i])
        min_low[i] = min(min_low[i-1], low[i])
    
    chop = np.zeros_like(close)
    for i in range(atr_period, len(close)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(sum(tr[i-atr_period+1:i+1]) / (atr_period * np.log10(max_high[i] - min_low[i])))
        else:
            chop[i] = 50
    
    # Volume confirmation: volume > 1.3x average volume (24-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=24, min_periods=24).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 24)  # for KAMA and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price below KAMA in choppy market with volume
            if (price < kama_aligned[i] and chop[i] > 61.8 and 
                vol > 1.3 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price above KAMA in choppy market with volume
            elif (price > kama_aligned[i] and chop[i] > 61.8 and 
                  vol > 1.3 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above KAMA OR chop decreases
            if price > kama_aligned[i] or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below KAMA OR chop decreases
            if price < kama_aligned[i] or chop[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Choppiness_KAMA_Reversal"
timeframe = "12h"
leverage = 1.0