#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, use KAMA to determine trend direction, RSI(2) for mean-reversion entries,
and Choppiness Index regime filter to avoid whipsaws. Enters long in uptrend when RSI<30 and CHOP>61.8 (range),
enters short in downtrend when RSI>70 and CHOP>61.8. Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Designed for 30-100 total trades over 4 years (7-25/year). Works in both bull and bear markets by adapting to regime.
"""

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
    
    # Load weekly data for HTF trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA on weekly close
    def calculate_kama(close_vals, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_vals, n=period))
        volatility = np.sum(np.abs(np.diff(close_vals)), axis=0)
        er = np.zeros_like(close_vals)
        er[period:] = change[period-1:] / volatility[period-1:]
        er[er < 0] = 0
        er[er > 1] = 1
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama = np.zeros_like(close_vals)
        kama[0] = close_vals[0]
        for i in range(1, len(close_vals)):
            kama[i] = kama[i-1] + sc[i] * (close_vals[i] - kama[i-1])
        return kama
    
    kama_1w = calculate_kama(df_1w['close'].values)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate RSI(2) on daily close
    def calculate_rsi(close_vals, period=2):
        delta = np.diff(close_vals)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_vals)
        avg_loss = np.zeros_like(close_vals)
        
        # Wilder's smoothing
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close_vals)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100
        rsi[:period] = np.nan
        return rsi
    
    rsi_2 = calculate_rsi(close)
    
    # Calculate Choppiness Index (14)
    def calculate_chop(high_vals, low_vals, close_vals, period=14):
        # True Range
        tr1 = high_vals[1:] - low_vals[1:]
        tr2 = np.abs(high_vals[1:] - close_vals[:-1])
        tr3 = np.abs(low_vals[1:] - close_vals[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr = np.zeros_like(close_vals)
        for i in range(1, len(close_vals)):
            if i < period:
                atr[i] = np.nan
            else:
                atr[i] = np.nanmean(tr[i-period+1:i+1])
        
        # Sum of ATR
        sum_atr = np.zeros_like(close_vals)
        for i in range(period-1, len(close_vals)):
            sum_atr[i] = np.nansum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close_vals)
        lowest_low = np.zeros_like(close_vals)
        for i in range(len(close_vals)):
            if i < period-1:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high_vals[i-period+1:i+1])
                lowest_low[i] = np.min(low_vals[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close_vals)
        for i in range(period-1, len(close_vals)):
            if sum_atr[i] > 0:
                chop[i] = 100 * np.log10(highest_high[i] - lowest_low[i]) / np.log10(period) / np.log10(sum_atr[i])
            else:
                chop[i] = np.nan
        chop[:period-1] = np.nan
        return chop
    
    chop = calculate_chop(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(30, 14)  # KAMA needs ~30, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_2[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine trend from weekly KAMA
        weekly_uptrend = close[i] > kama_1w_aligned[i]
        weekly_downtrend = close[i] < kama_1w_aligned[i]
        
        # Range regime: CHOP > 61.8
        is_range = chop[i] > 61.8
        
        # Long logic: weekly uptrend + RSI oversold + range regime
        if weekly_uptrend and rsi_2[i] < 30 and is_range:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: weekly downtrend + RSI overbought + range regime
        elif weekly_downtrend and rsi_2[i] > 70 and is_range:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: RSI returns to neutral or regime changes to trend
        elif position == 1 and (rsi_2[i] > 50 or chop[i] < 38.2):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi_2[i] < 50 or chop[i] < 38.2):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0