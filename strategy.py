#!/usr/bin/env python3
"""
12h_Wilson_RSI_Divergence_Trend
Hypothesis: RSI divergence on 12h with Wilson's oscillator confirmation for early trend detection. 
Wilson's oscillator (normalized MACD) provides momentum confirmation while RSI divergence signals exhaustion.
Designed for low trade frequency (15-25/year) with strong performance in both bull and bear markets by catching trend reversals early.
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
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Wilson's oscillator (normalized MACD-like)
    # Fast EMA(8) and Slow EMA(21) normalized by ATR(14)
    ema_fast = np.full(n, np.nan)
    ema_slow = np.full(n, np.nan)
    
    if n >= 8:
        ema_fast[7] = np.mean(close[0:8])
        alpha_fast = 2 / (8 + 1)
        for i in range(8, n):
            ema_fast[i] = close[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
    
    if n >= 21:
        ema_slow[20] = np.mean(close[0:21])
        alpha_slow = 2 / (21 + 1)
        for i in range(21, n):
            ema_slow[i] = close[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
    
    # ATR(14) for normalization
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    atr = np.full(n, np.nan)
    if n >= 14:
        atr[13] = np.mean(tr[1:15])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Wilson's oscillator
    wilson = np.full(n, np.nan)
    mask = (~np.isnan(ema_fast)) & (~np.isnan(ema_slow)) & (~np.isnan(atr)) & (atr != 0)
    wilson[mask] = (ema_fast[mask] - ema_slow[mask]) / atr[mask]
    
    # Wilson signal line (EMA of Wilson)
    wilson_signal = np.full(n, np.nan)
    if n >= 9:
        valid_start = np.where(~np.isnan(wilson))[0]
        if len(valid_start) > 0:
            start_idx = valid_start[0]
            if start_idx + 8 < n:
                wilson_signal[start_idx+8] = np.mean(wilson[start_idx:start_idx+9])
                alpha_signal = 2 / (9 + 1)
                for i in range(start_idx+9, n):
                    if not np.isnan(wilson[i]):
                        wilson_signal[i] = wilson[i] * alpha_signal + wilson_signal[i-1] * (1 - alpha_signal)
    
    # RSI divergence detection
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    
    def find_pivots(arr, lookback=5):
        """Find pivot points"""
        pivots = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr) - lookback):
            if np.isnan(arr[i]):
                continue
            # Check if it's a pivot low
            if all(arr[i] <= arr[i-j] for j in range(1, lookback+1)) and \
               all(arr[i] <= arr[i+j] for j in range(1, lookback+1)):
                pivots[i] = arr[i]  # pivot low
            # Check if it's a pivot high
            elif all(arr[i] >= arr[i-j] for j in range(1, lookback+1)) and \
                 all(arr[i] >= arr[i+j] for j in range(1, lookback+1)):
                pivots[i] = arr[i]  # pivot high
        return pivots
    
    rsi_pivots = find_pivots(rsi, 3)
    price_pivots_low = find_pivots(low, 3)
    price_pivots_high = find_pivots(high, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 21)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(wilson[i]) or np.isnan(wilson_signal[i]) or
            np.isnan(rsi_pivots[i]) or np.isnan(price_pivots_low[i]) or np.isnan(price_pivots_high[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for bullish divergence: price lower low, RSI higher low
            bullish_div = False
            bearish_div = False
            
            # Check last 10 bars for divergence
            lookback = min(10, i)
            if lookback >= 6:
                # Find recent pivot lows in price and RSI
                price_lows = []
                rsi_lows = []
                price_highs = []
                rsi_highs = []
                
                for j in range(i-lookback, i+1):
                    if not np.isnan(price_pivots_low[j]):
                        price_lows.append((j, price_pivots_low[j]))
                    if not np.isnan(rsi_pivots[j]) and not np.isnan(rsi[j]) and rsi[j] < 40:
                        rsi_lows.append((j, rsi[j]))
                    if not np.isnan(price_pivots_high[j]):
                        price_highs.append((j, price_pivots_high[j]))
                    if not np.isnan(rsi_pivots[j]) and not np.isnan(rsi[j]) and rsi[j] > 60:
                        rsi_highs.append((j, rsi[j]))
                
                # Check for bullish divergence
                if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                    price_lows_sorted = sorted(price_lows, key=lambda x: x[0])
                    rsi_lows_sorted = sorted(rsi_lows, key=lambda x: x[0])
                    if (price_lows_sorted[-1][0] > price_lows_sorted[-2][0] and 
                        rsi_lows_sorted[-1][0] > rsi_lows_sorted[-2][0]):
                        if (price_lows_sorted[-1][1] < price_lows_sorted[-2][1] and  # price lower low
                            rsi_lows_sorted[-1][1] > rsi_lows_sorted[-2][1]):      # RSI higher low
                            bullish_div = True
                
                # Check for bearish divergence
                if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                    price_highs_sorted = sorted(price_highs, key=lambda x: x[0])
                    rsi_highs_sorted = sorted(rsi_highs, key=lambda x: x[0])
                    if (price_highs_sorted[-1][0] > price_highs_sorted[-2][0] and 
                        rsi_highs_sorted[-1][0] > rsi_highs_sorted[-2][0]):
                        if (price_highs_sorted[-1][1] > price_highs_sorted[-2][1] and  # price higher high
                            rsi_highs_sorted[-1][1] < rsi_highs_sorted[-2][1]):      # RSI lower high
                            bearish_div = True
            
            # Entry conditions: divergence + Wilson crossover
            if bullish_div and wilson[i] > wilson_signal[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_div and wilson[i] < wilson_signal[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or Wilson bearish crossover
            if rsi[i] > 70 or wilson[i] < wilson_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or Wilson bullish crossover
            if rsi[i] < 30 or wilson[i] > wilson_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Wilson_RSI_Divergence_Trend"
timeframe = "12h"
leverage = 1.0