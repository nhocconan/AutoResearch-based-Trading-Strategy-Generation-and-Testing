#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI mean reversion + chop regime filter
# KAMA adapts to market noise, reducing false signals in choppy markets.
# RSI > 60 or < 40 provides entry signals in trending markets.
# Chop filter (Choppiness Index > 61.8) avoids trend-following in ranging markets.
# Target: 20-30 trades per year (80-120 total over 4 years) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, window=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        for i in range(window, len(close)):
            atr[i] = np.mean(tr[i-window+1:i+1])
        
        sum_atr = np.zeros(len(close))
        for i in range(window, len(close)):
            sum_atr[i] = np.sum(atr[i-window+1:i+1])
        
        max_range = np.zeros(len(close))
        min_range = np.zeros(len(close))
        for i in range(window, len(close)):
            max_range[i] = np.max(high[i-window+1:i+1])
            min_range[i] = np.min(low[i-window+1:i+1])
        
        chop = np.zeros(len(close))
        for i in range(window, len(close)):
            if max_range[i] != min_range[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_range[i] - min_range[i])) / np.log10(window)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama_vals = kama(close)
    rsi = np.zeros_like(close)
    # RSI calculation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(14, len(close)):
        avg_gain[i] = np.mean(gain[i-13:i+1])
        avg_loss[i] = np.mean(loss[i-13:i+1])
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    chop_vals = choppiness_index(high, low, close)
    
    # Weekly trend filter
    close_1w = df_1w['close'].values
    sma_1w = np.zeros(len(close_1w))
    for i in range(len(close_1w)):
        if i < 20:
            sma_1w[i] = np.mean(close_1w[:i+1])
        else:
            sma_1w[i] = np.mean(close_1w[i-19:i+1])
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_vals[i]) or np.isnan(sma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_vals[i]
        rsi_val = rsi[i]
        chop_val = chop_vals[i]
        weekly_sma = sma_1w_aligned[i]
        
        # Chop regime filter: only trend-follow when chop < 61.8 (trending market)
        # In choppy markets (chop > 61.8), use mean reversion
        if chop_val < 61.8:
            # Trending market: follow KAMA direction
            if position == 0:
                if price > kama_val and price > weekly_sma:
                    position = 1
                    signals[i] = position_size
                elif price < kama_val and price < weekly_sma:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                if price < kama_val:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                if price > kama_val:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:
            # Choppy market: mean reversion using RSI
            if position == 0:
                if rsi_val < 40 and price > kama_val:  # Oversold + above KAMA
                    position = 1
                    signals[i] = position_size
                elif rsi_val > 60 and price < kama_val:  # Overbought + below KAMA
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif position == 1:
                if rsi_val > 60 or price < kama_val:  # Overbought or below KAMA
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            elif position == -1:
                if rsi_val < 40 or price > kama_val:  # Oversold or above KAMA
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
    
    return signals

name = "1d_1w_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0