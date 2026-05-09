#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI + chop filter.
# Uses Kaufman Adaptive Moving Average (KAMA) to identify trend direction.
# Enters long when KAMA up + RSI < 40 (oversold bounce) + chop > 61.8 (range).
# Enters short when KAMA down + RSI > 60 (overbought pullback) + chop > 61.8 (range).
# Chop filter avoids whipsaw in strong trends, focusing on mean reversion in ranges.
# Designed for mean reversion in ranging markets while avoiding strong trends.
# Works in both bull and bear by following KAMA direction and fading extremes.
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(price, period=10, fast=2, slow=30):
        """Kaufman Adaptive Moving Average"""
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=0)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_val = np.zeros_like(price)
        kama_val[:period] = price[:period]
        for i in range(period, len(price)):
            kama_val[i] = kama_val[i-1] + sc[i] * (price[i] - kama_val[i-1])
        return kama_val
    
    kama_val = kama(close, 10, 2, 30)
    
    # RSI (Relative Strength Index)
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_val = rsi(close, 14)
    
    # Choppiness Index
    def choppiness_index(high, low, close, period=14):
        """Choppiness Index: measures whether market is choppy (ranging) or trending"""
        atr = np.zeros(len(close))
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        # True Range alternative using pandas for efficiency
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop_val = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(chop_val[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up (trend up) + RSI oversold (<40) + choppy market (>61.8)
            if (i > 0 and kama_val[i] > kama_val[i-1] and 
                rsi_val[i] < 40 and chop_val[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (trend down) + RSI overbought (>60) + choppy market (>61.8)
            elif (i > 0 and kama_val[i] < kama_val[i-1] and 
                  rsi_val[i] > 60 and chop_val[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA down or RSI overbought (>70) or chop low (<38.2 = trending)
            if (kama_val[i] < kama_val[i-1] or 
                rsi_val[i] > 70 or chop_val[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA up or RSI oversold (<30) or chop low (<38.2 = trending)
            if (kama_val[i] > kama_val[i-1] or 
                rsi_val[i] < 30 or chop_val[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals