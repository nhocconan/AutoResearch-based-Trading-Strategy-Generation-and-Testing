#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA (Kaufman Adaptive Moving Average) with RSI and Choppiness regime filter
# Uses 1d KAMA for trend direction, 4h RSI for entry timing, and 1d Choppiness Index for regime detection
# Long in trending regimes when price > KAMA and RSI > 50; Short when price < KAMA and RSI < 50
# Range-bound regimes avoid trading to reduce whipsaw
# Target: 20-35 trades/year per symbol, works in bull/bear via trend and regime filters

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for KAMA and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 1 else np.sum(np.abs(np.diff(close)))
        er = np.zeros_like(change)
        er[length:] = change[length:] / (volatility[length:] + 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1d = calculate_kama(close_1d, length=10, fast=2, slow=30)
    
    # Calculate 1-day Choppiness Index
    def calculate_choppiness(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Sum of true ranges over period
        atr_sum = np.zeros_like(close)
        for i in range(length, len(close)):
            atr_sum[i] = np.sum(tr[i-length+1:i+1])
        # True range
        true_range = tr
        # Choppiness formula: 100 * log10(atr_sum / (true_range * length)) / log10(length)
        chop = np.full_like(close, 50.0)
        for i in range(length, len(close)):
            if true_range[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (true_range[i] * length)) / np.log10(length)
        return chop
    
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, length=14)
    
    # Calculate 4-hour RSI (14-period)
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        for i in range(1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, length=14)
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: Choppiness > 61.8 = ranging (avoid trade), < 38.2 = trending
        is_trending = chop_1d_aligned[i] < 38.2
        
        if position == 0 and is_trending:
            # Long: Price > KAMA + RSI > 50 + volume spike
            if (close[i] > kama_1d_aligned[i] and rsi[i] > 50 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA + RSI < 50 + volume spike
            elif (close[i] < kama_1d_aligned[i] and rsi[i] < 50 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse signal or exit ranging market
            if position == 1:
                if (close[i] < kama_1d_aligned[i] or rsi[i] < 50 or not is_trending):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > kama_1d_aligned[i] or rsi[i] > 50 or not is_trending):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Chop_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0