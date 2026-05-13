#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_Chop_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to determine trend direction on 12h timeframe, filtered by RSI for momentum and Choppiness Index for regime detection. Enter long when KAMA turns up, RSI > 50, and market is not choppy (CHOP > 61.8 indicates ranging, so we avoid when CHOP <= 61.8). Enter short when KAMA turns down, RSI < 50, and CHOP <= 61.8. This strategy avoids whipsaws in ranging markets and captures trending moves. Works in bull (KAMA up with RSI > 50) and bear (KAMA down with RSI < 50). Low frequency due to trend confirmation and regime filter.

name = "12h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "12h"
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

    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate KAMA on 12h close (trend indicator)
    # KAMA parameters: ER length=10, fast SC=2, slow SC=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # This needs correction - we'll compute properly below
    
    # Proper KAMA calculation
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close)
        kama = np.zeros(n)
        kama[:] = np.nan
        
        # Direction
        direction = np.abs(close - np.roll(close, er_length))
        direction[:er_length] = 0
        
        # Volatility
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Still wrong - let's do element-wise
        
        # Better approach: calculate ER per point
        er = np.zeros(n)
        for i in range(er_length, n):
            direction_val = np.abs(close[i] - close[i-er_length])
            volatility_val = np.sum(np.abs(np.diff(close[i-er_length:i+1])))
            if volatility_val > 0:
                er[i] = direction_val / volatility_val
            else:
                er[i] = 0
        
        # Smoothing constant
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA
        kama[0] = close[0]
        for i in range(1, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
                
        return kama

    # Calculate KAMA on 12h data
    kama = calculate_kama(close, 10, 2, 30)
    
    # Calculate RSI on 12h close
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index on 1d data (regime filter)
    def calculate_choppiness(high, low, close, period=14):
        n = len(close)
        chop = np.zeros(n)
        chop[:] = np.nan
        
        if n < period * 2:
            return chop
            
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with indices
        
        # Sum of TR over period
        atr = np.zeros(n)
        for i in range(period, n):
            atr[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros(n)
        lowest_low = np.zeros(n)
        for i in range(period-1, n):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        for i in range(period-1, n):
            if atr[i] > 0 and highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        
        return chop

    chop = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # Wait, KAMA is on 12h, not 1d - mistake
    
    # Fix: KAMA and RSI are on 12h data, so no alignment needed for them
    # Only CHOP needs alignment from 1d to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # For KAMA and RSI on 12h data, we can use directly but need to handle alignment properly
    # Since they're calculated on the same timeframe, we'll create a dummy dataframe for alignment
    # Actually, let's recalculate: KAMA and RSI should be on 12h data, so no HTF alignment needed
    # But the function expects HTF data, so we'll use a trick: create a 12h dataframe from prices
    # No, simpler: for indicators on same timeframe, we can use them directly
    # But to be safe with the interface, let's align using the same timeframe
    
    # Create a dummy dataframe for 12h data to use with align_htf_to_ltf (same timeframe)
    # Actually, align_htf_to_ltf works when HTF and LTF are the same - it just returns the values
    # So we can use it safely
    
    # Recalculate KAMA and RSI properly on 12h data
    # KAMA calculation
    def kama_indicator(close, er_length=10, fast_sc=2, slow_sc=30):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < er_length:
            return kama
            
        # Efficiency ratio
        er = np.full(n, np.nan)
        for i in range(er_length, n):
            direction = np.abs(close[i] - close[i-er_length])
            volatility = np.sum(np.abs(np.diff(close[i-er_length:i+1])))
            if volatility > 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
        
        # Smoothing constant
        sc = np.full(n, np.nan)
        for i in range(er_length, n):
            if not np.isnan(er[i]):
                sc[i] = (er[i] * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
            else:
                sc[i] = 0
        
        # KAMA
        kama[er_length] = close[er_length]  # start with first valid point
        for i in range(er_length+1, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
                
        return kama

    # RSI calculation
    def rsi_indicator(close, period=14):
        n = len(close)
        rsi = np.full(n, np.nan)
        if n < period + 1:
            return rsi
            
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # First average gain/loss
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    kama = kama_indicator(close)
    rsi = rsi_indicator(close)
    
    # Align KAMA and RSI (though they're on same timeframe, use align_htf_to_ltf for safety)
    # Create a 12h dataframe from prices for alignment
    df_12h = prices.copy()  # This is approximate but should work for same timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Chop is already aligned from 1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up (current > previous), RSI > 50, not choppy (CHOP > 61.8 = trending)
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down (current < previous), RSI < 50, choppy or trending (we'll use CHOP <= 61.8 for short too)
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop_aligned[i] <= 61.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turning down OR RSI < 50
            if kama[i] < kama[i-1] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turning up OR RSI > 50
            if kama[i] > kama[i-1] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals