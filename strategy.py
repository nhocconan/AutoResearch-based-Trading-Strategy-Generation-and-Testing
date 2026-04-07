#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# Combined with RSI for momentum and Choppiness Index to filter ranging markets, this strategy avoids whipsaws.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Target: 15-25 trades/year (60-100 total over 4 years).

name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - faster adaptation to trend changes
    def calculate_kama(close, slow=10, fast=2):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=1))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(change) > 0 else 0
        er = np.zeros_like(close)
        er[1:] = change / (volatility + 1e-10)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA
    kama = calculate_kama(close, slow=10, fast=2)
    
    # RSI (Relative Strength Index)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Choppiness Index (to filter ranging markets)
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
            atr[i] = (atr[i-1] * (period-1) + tr) / period if i > 0 else tr
        # Sum of true ranges over period
        sum_tr = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_tr[i] = np.sum(atr[i-period+1:i+1])
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(len(close)):
            if i >= period:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if sum_tr[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(sum_tr[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # Neutral when undefined
        return chop
    
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA or chop too high (ranging market)
            if close[i] < kama[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above KAMA or chop too high (ranging market)
            if close[i] > kama[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade when market is trending (chop < 61.8)
            if chop[i] < 61.8:
                # Go long if price above KAMA and RSI > 50 (bullish momentum)
                if close[i] > kama[i] and rsi[i] > 50:
                    # Additional weekly trend filter: only long if above weekly EMA
                    if close[i] > ema_20_1w_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                # Go short if price below KAMA and RSI < 50 (bearish momentum)
                elif close[i] < kama[i] and rsi[i] < 50:
                    # Additional weekly trend filter: only short if below weekly EMA
                    if close[i] < ema_20_1w_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals