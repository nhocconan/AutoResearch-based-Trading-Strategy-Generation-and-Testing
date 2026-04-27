#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI and Chop filter.
# Uses KAMA to detect trend direction, RSI for mean-reversion entries, and Chop index for regime filtering.
# In trending markets (Chop < 38.2), we take KAMA trend continuation with RSI pullbacks.
# In ranging markets (Chop > 61.8), we take mean-reversion at RSI extremes.
# Designed to work in both bull and bear markets with low trade frequency.
# Target: 15-25 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on close
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0])).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum().shift(1)
        volatility = np.concatenate([[0], volatility[:-1]])  # shift right
        
        er = np.zeros_like(close)
        for i in range(er_length, len(close)):
            if np.sum(volatility[i-er_length+1:i+1]) > 0:
                er[i] = np.abs(close[i] - close[i-er_length]) / np.sum(volatility[i-er_length+1:i+1])
            else:
                er[i] = 0
        
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        for i in range(1, len(close)):
            if i < length:
                avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
                avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
            else:
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_vals = rsi(close, 14)
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            if i < length:
                atr[i] = np.mean(atr[1:i+1]) if i > 0 else tr
            else:
                atr[i] = (atr[i-1] * (length-1) + tr) / length
        
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            highest_high[i] = np.max(high[i-length+1:i+1])
            lowest_low[i] = np.min(low[i-length+1:i+1])
        
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if highest_high[i] - lowest_low[i] > 0:
                chop[i] = 100 * np.log10(np.sum(atr[i-length+1:i+1]) / (highest_high[i] - lowest_low[i])) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    chop_vals = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend from KAMA slope
        if i > 0:
            kama_rising = kama_vals[i] > kama_vals[i-1]
            kama_falling = kama_vals[i] < kama_vals[i-1]
        else:
            kama_rising = False
            kama_falling = False
        
        if position == 0:
            # Trending market: Chop < 38.2
            if chop_vals[i] < 38.2:
                # Long: KAMA rising + RSI pullback from overbought
                if kama_rising and rsi_vals[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: KAMA falling + RSI pullback from oversold
                elif kama_falling and rsi_vals[i] > 60:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: Chop > 61.8
            elif chop_vals[i] > 61.8:
                # Long at RSI oversold
                if rsi_vals[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Short at RSI overbought
                elif rsi_vals[i] > 70:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: KAMA turns down or RSI overbought in trend, or RSI extreme in range
            if (chop_vals[i] < 38.2 and (not kama_rising or rsi_vals[i] > 70)) or \
               (chop_vals[i] > 61.8 and rsi_vals[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up or RSI oversold in trend, or RSI extreme in range
            if (chop_vals[i] < 38.2 and (not kama_falling or rsi_vals[i] < 30)) or \
               (chop_vals[i] > 61.8 and rsi_vals[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_System_v1"
timeframe = "1d"
leverage = 1.0