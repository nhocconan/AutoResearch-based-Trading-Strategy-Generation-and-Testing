# 1d_KAMA_RSI_ChopFilter
# Hypothesis: 1d KAMA direction with RSI momentum and Choppiness index regime filter
# KAMA adapts to market efficiency - trend following in trending markets, mean-reverting in choppy
# RSI(14) > 50 for long momentum, < 50 for short momentum
# Choppiness index > 61.8 indicates ranging market (avoid trend following)
# Choppiness index < 38.2 indicates trending market (follow trend)
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in bull/bear via adaptive KAMA and regime filter

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # KAMA (Kaufman Adaptive Moving Average) - 1d
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_length))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
        # For array calculation
        er = np.zeros_like(close)
        for i in range(er_length, len(close)):
            price_change = np.abs(close[i] - close[i-er_length])
            sum_abs_diff = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
            if sum_abs_diff != 0:
                er[i] = price_change / sum_abs_diff
            else:
                er[i] = 0
        er[:er_length] = 0
        
        # Smoothing Constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI
    def calculate_rsi(close, period=14):
        delta = np.diff(close, n=1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(close) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Wilder smoothing
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index
    def calculate_choppiness(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Sum of TR over period
        atr_sum = np.zeros_like(close)
        for i in range(period-1, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Choppiness formula
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    # Calculate indicators on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA
    kama_1d = calculate_kama(df_1d['close'].values)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI
    rsi_1d = calculate_rsi(df_1d['close'].values)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Choppiness Index
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        trending_market = chop_1d_aligned[i] < 38.2
        
        if position == 0:
            # Long: price above KAMA and RSI > 50 in trending market
            if (close[i] > kama_1d_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                trending_market):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI < 50 in trending market
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  trending_market):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price below KAMA or RSI < 45 or market becomes choppy
            if (close[i] < kama_1d_aligned[i]) or (rsi_1d_aligned[i] < 45) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price above KAMA or RSI > 55 or market becomes choppy
            if (close[i] > kama_1d_aligned[i]) or (rsi_1d_aligned[i] > 55) or (chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals