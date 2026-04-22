# 1d_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: 1-day KAMA trend with RSI momentum filter and Choppiness regime filter.
# Long when KAMA trend is up, RSI > 50, and Choppiness < 38.2 (trending market).
# Short when KAMA trend is down, RSI < 50, and Choppiness < 38.2.
# Exit when KAMA trend reverses or Choppiness > 61.8 (choppy market).
# Uses daily timeframe for lower trade frequency and better generalization in bull/bear markets.
# KAMA adapts to market efficiency, reducing whipsaws in sideways markets.
# RSI filters for momentum strength, Choppiness avoids ranging conditions.
# Expected trades: 15-25 per year (60-100 total over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on close
    # Fast EMA period = 2, Slow EMA period = 30
    # ER = Efficiency Ratio, SC = Smoothing Constant
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate change and volatility for ER
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder for loop
    
    # Calculate ER and KAMA in loop for clarity and correctness
    er = np.zeros(n)
    sc = np.zeros(n)
    kama = np.zeros(n)
    
    # Initialize
    kama[0] = close[0]
    
    for i in range(1, n):
        # Directional change
        directional_change = np.abs(close[i] - close[i-9]) if i >= 9 else np.abs(close[i] - close[0])
        # Volatility (sum of absolute changes over 10 periods)
        volatility_sum = np.sum(np.abs(np.diff(close[max(0, i-9):i+1]))) if i >= 1 else 0
        # Avoid division by zero
        er[i] = directional_change / volatility_sum if volatility_sum > 0 else 0
        # Smoothing constant
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        # KAMA
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.zeros(n)
    
    # Initialize first average
    avg_gain[13] = np.mean(gain[1:14]) if n > 13 else 0
    avg_loss[13] = np.mean(loss[1:14]) if n > 13 else 0
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 100
        rsi[i] = 100 - (100 / (1 + rs))
    
    # For periods before 14, set RSI to 50 (neutral)
    rsi[:14] = 50
    
    # Calculate Choppiness Index (14-period) - needs high, low, close
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first period
    
    # Sum of True Range over 14 periods
    atr_sum = np.zeros(n)
    for i in range(14, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    # For first 14 periods, use cumulative sum
    for i in range(1, 14):
        atr_sum[i] = atr_sum[i-1] + tr[i]
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i-13)
        max_high[i] = np.max(high[start_idx:i+1])
        min_low[i] = np.min(low[start_idx:i+1])
    
    # Avoid division by zero
    range_hl = max_high - min_low
    chop = np.zeros(n)
    for i in range(14, n):
        if range_hl[i] > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    # For first 14 periods
    chop[:14] = 50
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if any data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA trend: price above/below KAMA
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI conditions
        rsi_above_50 = rsi[i] > 50
        rsi_below_50 = rsi[i] < 50
        
        # Choppiness regime: trending when CHOP < 38.2, choppy when CHOP > 61.8
        trending_market = chop[i] < 38.2
        choppy_market = chop[i] > 61.8
        
        if position == 0:
            # Enter long: KAMA up, RSI > 50, trending market
            if kama_up and rsi_above_50 and trending_market:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA down, RSI < 50, trending market
            elif kama_down and rsi_below_50 and trending_market:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA down OR choppy market
                if not kama_up or choppy_market:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA up OR choppy market
                if not kama_down or choppy_market:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_and_Chop_Filter"
timeframe = "1d"
leverage = 1.0