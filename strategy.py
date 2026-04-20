#!/usr/bin/env python3
"""
4h_KAMA_Direction_Plus_RSI_With_Chop_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. RSI identifies overbought/oversold conditions. Choppiness filter avoids whipsaw in ranging markets. Works in both bull and bear by only trading when trend is strong (KAMA aligned) and market is not choppy.
"""

name = "4h_KAMA_Direction_Plus_RSI_With_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (adaptive moving average)
    def calculate_kama(close, er_len=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_len))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, n=1)).sum()
        # Manual volatility sum for rolling window
        volatility_sum = np.zeros_like(close)
        for i in range(er_len, len(close)):
            volatility_sum[i] = np.sum(np.abs(np.diff(close[i-er_len:i])))
        volatility_sum[:er_len] = np.nan
        er = np.where(volatility_sum != 0, change / volatility_sum, 0)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.mean(data[:period])
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        up_smoothed = wilder_smooth(up, period)
        down_smoothed = wilder_smooth(down, period)
        rs = np.where(down_smoothed != 0, up_smoothed / down_smoothed, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR
        atr = np.zeros_like(close)
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period if i >= 1 else tr[i]
        atr[:period-1] = np.nan
        
        # Sum of TR over period
        tr_sum = np.zeros_like(close)
        for i in range(period-1, len(tr)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        tr_sum[:period-1] = np.nan
        
        # Max and min over period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(period-1, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        max_high[:period-1] = np.nan
        min_low[:period-1] = np.nan
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, 10, 2, 30)
    rsi = calculate_rsi(close, 14)
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA AND RSI < 50 (not overbought) AND Chop < 61.8 (not choppy)
            if close[i] > kama[i] and rsi[i] < 50 and chop_1d_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA AND RSI > 50 (not oversold) AND Chop < 61.8 (not choppy)
            elif close[i] < kama[i] and rsi[i] > 50 and chop_1d_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below KAMA OR RSI > 70 (overbought) OR Chop > 61.8 (choppy)
            if close[i] < kama[i] or rsi[i] > 70 or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above KAMA OR RSI < 30 (oversold) OR Chop > 61.8 (choppy)
            if close[i] > kama[i] or rsi[i] < 30 or chop_1d_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals