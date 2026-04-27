#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime strategy
# Uses Kaufman's Adaptive Moving Average for trend direction, RSI for momentum,
# and Choppiness Index to filter regimes (choppy = mean reversion, trending = follow trend).
# Works in bull/bear by adapting to market regime via Chop filter.
# Target: 30-100 total trades over 4 years (~7-25/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (10-period ER, 2/30 fast/slow SC)
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        n = len(close_prices)
        kama = np.full(n, np.nan)
        if n < length:
            return kama
        
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, n=length))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        er = np.zeros(n)
        er[length:] = change[length-1:] / np.maximum(volatility[length-1:], 1e-10)
        
        # Smoothing Constant
        sc = np.zeros(n)
        sc[length:] = (er[length:] * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama[length-1] = close_prices[length-1]
        for i in range(length, n):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    kama_10 = calculate_kama(close_1d, 10, 2, 30)
    kama_10_aligned = align_htf_to_ltf(prices, df_1d, kama_10)
    
    # Calculate RSI (14-period)
    def calculate_rsi(close_prices, length=14):
        n = len(close_prices)
        rsi = np.full(n, np.nan)
        if n < length + 1:
            return rsi
        
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        
        for i in range(length + 1, n):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi = 100 - (100 / (1 + rs))
        rsi[length:] = 100 - (100 / (1 + rs[length:]))
        return rsi
    
    rsi_14 = calculate_rsi(close_1d, 14)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate Choppiness Index (14-period)
    def calculate_chop(high_prices, low_prices, close_prices, length=14):
        n = len(close_prices)
        chop = np.full(n, np.nan)
        if n < length:
            return chop
        
        atr = np.zeros(n)
        for i in range(1, n):
            atr[i] = max(
                high_prices[i] - low_prices[i],
                abs(high_prices[i] - close_prices[i-1]),
                abs(low_prices[i] - close_prices[i-1])
            )
        
        for i in range(length, n):
            atr_sum = np.sum(atr[i-length+1:i+1])
            highest_high = np.max(high_prices[i-length+1:i+1])
            lowest_low = np.min(low_prices[i-length+1:i+1])
            if atr_sum > 0:
                chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    chop_14 = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_14_aligned = align_htf_to_ltf(prices, df_1d, chop_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA (10), RSI (14), CHOP (14)
    start_idx = max(10, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_10_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(chop_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama = kama_10_aligned[i]
        rsi = rsi_14_aligned[i]
        chop = chop_14_aligned[i]
        
        # Regime filter: Chop > 61.8 = ranging (mean revert), Chop < 38.2 = trending
        is_ranging = chop > 61.8
        is_trending = chop < 38.2
        
        if position == 0:
            # Long conditions
            if is_ranging:
                # Mean reversion in ranging market: buy oversold
                if price < kama and rsi < 30:
                    signals[i] = size
                    position = 1
            else:  # trending
                # Trend following: buy in uptrend
                if price > kama and rsi > 50:
                    signals[i] = size
                    position = 1
            # Short conditions
            if is_ranging:
                # Mean reversion in ranging market: sell overbought
                if price > kama and rsi > 70:
                    signals[i] = -size
                    position = -1
            else:  # trending
                # Trend following: sell in downtrend
                if price < kama and rsi < 50:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: opposite conditions
            if is_ranging:
                # Exit mean reversion when price returns to KAMA or RSI normalizes
                if price >= kama or rsi >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                # Exit trend when price crosses KAMA or RSI weakens
                if price <= kama or rsi < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Exit short: opposite conditions
            if is_ranging:
                # Exit mean reversion when price returns to KAMA or RSI normalizes
                if price <= kama or rsi >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                # Exit trend when price crosses KAMA or RSI weakens
                if price >= kama or rsi > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0