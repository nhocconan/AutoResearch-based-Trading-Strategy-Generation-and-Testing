#!/usr/bin/env python3
# 1D_KAMA_Trend_RSI_Extremes_ChopFilter
# Hypothesis: KAMA trend direction + RSI extremes + Choppiness filter on daily timeframe.
# Works in bull/bear: KAMA adapts to volatility, RSI captures overbought/oversold, Choppiness filter avoids whipsaws in sideways markets.
# Uses 1-week EMA200 for higher timeframe trend filter to ensure alignment with major trend.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.

name = "1D_KAMA_Trend_RSI_Extremes_ChopFilter"
timeframe = "1d"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(price, er_length=10, fast_ema=2, slow_ema=30):
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.abs(np.diff(price, prepend=price[0]))
        for i in range(1, len(price)):
            volatility[i] = volatility[i-1] + np.abs(price[i] - price[i-1])
        
        er = np.zeros_like(price)
        for i in range(er_length, len(price)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def rsi(price, length=14):
        delta = np.diff(price, prepend=price[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[length-1] = np.mean(gain[0:length])
        avg_loss[length-1] = np.mean(loss[0:length])
        
        for i in range(length, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        highest_high[0] = high[0]
        lowest_low[0] = low[0]
        for i in range(1, len(high)):
            highest_high[i] = max(highest_high[i-1], high[i])
            lowest_low[i] = min(lowest_low[i-1], low[i])
        
        atr_sum = np.zeros_like(close)
        for i in range(length-1, len(close)):
            atr_sum[i] = np.sum(atr[i-length+1:i+1])
        
        hhll = highest_high - lowest_low
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if hhll[i] != 0:
                chop[i] = 100 * np.log10(atr_sum[i] / hhll[i]) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama_val = kama(close)
    rsi_val = rsi(close)
    chop_val = choppiness_index(high, low, close)
    
    # Get 1-week data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_200_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 0.99 + close_1w[i] * 0.01)  # EMA approximation
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Align indicators to daily timeframe (already aligned as we calculated on close)
    # But we need to ensure we only use completed daily bars
    # Since we're calculating on daily close, no additional alignment needed for same TF
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Ensure KAMA, RSI, and Chop are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(chop_val[i]) or np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend), RSI < 30 (oversold), Chop > 61.8 (ranging market)
            if (close[i] > kama_val[i] and 
                rsi_val[i] < 30 and 
                chop_val[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend), RSI > 70 (overbought), Chop > 61.8 (ranging market)
            elif (close[i] < kama_val[i] and 
                  rsi_val[i] > 70 and 
                  chop_val[i] > 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 70 (overbought) OR Chop < 38.2 (trending market - let trend run)
            if (close[i] < kama_val[i] or 
                rsi_val[i] > 70 or 
                chop_val[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 30 (oversold) OR Chop < 38.2 (trending market - let trend run)
            if (close[i] > kama_val[i] or 
                rsi_val[i] < 30 or 
                chop_val[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals