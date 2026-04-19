#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI mean reversion + chop filter
# In trending markets (chop < 61.8), trade in direction of KAMA
# In ranging markets (chop > 61.8), use RSI extremes for mean reversion
# Works in both bull and bear by adapting to volatility regime
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years)

name = "1d_KAMA_RSI_Chop"
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
    
    # Calculate KAMA on close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # First 10 values invalid
    
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # For each point, sum of absolute changes over last 10 periods
    volatility_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start at index 9
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, np.nan)  # Align with close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.nanmean(gain[1:14])  # First average
    avg_loss[13] = np.nanmean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - np.roll(close, 1)[1:])
    tr3 = np.abs(low[1:] - np.roll(close, 1)[1:])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, np.nan)  # Align with close
    
    # ATR(14)
    atr = np.zeros_like(close)
    for i in range(14, len(close)):
        atr[i] = np.nansum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(14, len(close)):
        hh[i] = np.nanmax(high[i-13:i+1])
        ll[i] = np.nanmin(low[i-13:i+1])
    
    # Chop calculation
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        if atr[i] > 0 and (hh[i] - ll[i]) > 0:
            chop[i] = 100 * np.log10(np.nansum(tr[i-13:i+1]) / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # Default when undefined
    
    # Chop < 61.8 = trending, Chop > 61.8 = ranging
    chop_threshold = 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(close[i-1])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_close = close[i-1]
        chop_val = chop[i]
        
        if chop_val < chop_threshold:  # Trending market
            # Trade in direction of KAMA
            if price > kama[i] and prev_close <= kama[i-1]:
                # Bullish crossover
                signals[i] = 0.25
                position = 1
            elif price < kama[i] and prev_close >= kama[i-1]:
                # Bearish crossover
                signals[i] = -0.25
                position = -1
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25 if position == -1 else 0.0
        else:  # Ranging market
            # Mean reversion using RSI extremes
            if rsi[i] < 30 and position <= 0:
                # Oversold - go long
                signals[i] = 0.25
                position = 1
            elif rsi[i] > 70 and position >= 0:
                # Overbought - go short
                signals[i] = -0.25
                position = -1
            elif (rsi[i] > 40 and rsi[i] < 60) and position != 0:
                # Exit when RSI returns to neutral
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25 if position == -1 else 0.0
    
    return signals