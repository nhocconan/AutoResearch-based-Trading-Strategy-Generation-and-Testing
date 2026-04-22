#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA direction + RSI + chop filter on 1d timeframe
    # KAMA adapts to market noise, providing reliable trend direction
    # RSI(14) with overbought/oversold levels for mean reversion in chop
    # Choppiness index filter to avoid whipsaw in strong trends
    # Works in bull/bear: trend follow in trends, mean revert in ranges
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # For efficiency ratio over 10 periods
    er = np.zeros_like(change)
    for i in range(len(change)):
        if i >= 9:
            price_change = np.abs(close[i] - close[i-9])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 0
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = pd.Series(np.maximum(high - low, 
                               np.maximum(np.abs(high - np.roll(close, 1)),
                                          np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr[0] = high[0] - low[0]
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(14)
    # Handle edge cases where highest_high == lowest_low
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(14, n):  # Start after warmup for indicators
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # In choppy market (CHOP > 61.8), mean revert at RSI extremes
            if chop[i] > 61.8:
                if rsi[i] < 30 and close[i] > kama[i]:  # Oversold and price above KAMA
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and close[i] < kama[i]:  # Overbought and price below KAMA
                    signals[i] = -0.25
                    position = -1
            # In trending market (CHOP < 38.2), follow KAMA direction
            else:
                if close[i] > kama[i]:  # Uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i]:  # Downtrend
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:  # Long position
                # Exit if: RSI overbought OR trend change (price below KAMA in chop) OR chop becomes too high
                if rsi[i] > 70 or (chop[i] > 61.8 and close[i] < kama[i]) or (chop[i] < 38.2 and close[i] < kama[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit if: RSI oversold OR trend change (price above KAMA in chop) OR chop becomes too high
                if rsi[i] < 30 or (chop[i] > 61.8 and close[i] > kama[i]) or (chop[i] < 38.2 and close[i] > kama[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0