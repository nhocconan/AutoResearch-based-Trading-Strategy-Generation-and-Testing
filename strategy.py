#!/usr/bin/env python3
name = "12h_KAMA_RSI_Chop_Filter_v1"
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
    
    # Load 1D data ONCE for KAMA, RSI, and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA calculation (ER = 10)
    def calculate_kama(close, er_period=10):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # Handle edge cases
        er = np.zeros_like(close)
        er[:] = np.nan
        for i in range(er_period, len(close)):
            if volatility[i-er_period:i].sum() != 0:
                er[i] = change[i] / volatility[i-er_period:i].sum()
            else:
                er[i] = 0
        sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
        kama = np.full_like(close, np.nan)
        kama[er_period] = close[er_period]
        for i in range(er_period + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        if len(gain) >= period:
            avg_gain[period-1] = np.nanmean(gain[1:period])
            avg_loss[period-1] = np.nanmean(loss[1:period])
            
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.full_like(close, np.nan)
        valid = ~np.isnan(avg_loss) & (avg_loss != 0)
        rs[valid] = avg_gain[valid] / avg_loss[valid]
        
        rsi = np.full_like(close, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index calculation
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        highest_high = np.zeros_like(high)
        lowest_low = np.zeros_like(low)
        for i in range(len(close)):
            if i < period:
                highest_high[i] = np.max(high[:i+1])
                lowest_low[i] = np.min(low[:i+1])
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        sum_atr = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                sum_atr[i] = np.sum(atr[:i+1])
            else:
                sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        chop = np.full_like(close, 50.0)
        for i in range(period-1, len(close)):
            if highest_high[i] != lowest_low[i] and sum_atr[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    # Calculate indicators on 1D
    kama_1d = calculate_kama(close_1d, 10)
    rsi_1d = calculate_rsi(close_1d, 14)
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1D indicators to 12H timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after sufficient data
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope of KAMA
        if i >= 31:
            kama_slope = kama_1d_aligned[i] - kama_1d_aligned[i-1]
        else:
            kama_slope = 0
        
        # Chop filter: range-bound market (chop > 61.8) for mean reversion
        is_ranging = chop_1d_aligned[i] > 61.8
        
        # RSI for mean reversion in ranging markets
        rsi_oversold = rsi_1d_aligned[i] < 35
        rsi_overbought = rsi_1d_aligned[i] > 65
        
        if position == 0:
            # LONG: Ranging market + RSI oversold
            if is_ranging and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # SHORT: Ranging market + RSI overbought
            elif is_ranging and rsi_overbought:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or chop drops (trending)
            if rsi_1d_aligned[i] > 65 or chop_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or chop drops (trending)
            if rsi_1d_aligned[i] < 35 or chop_1d_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals