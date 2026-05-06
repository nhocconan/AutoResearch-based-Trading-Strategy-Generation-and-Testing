#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime filter
# Uses KAMA (Kaufman Adaptive Moving Average) on 1d for trend direction
# RSI(14) for momentum confirmation with overbought/oversold levels
# Choppiness Index(14) to detect ranging markets (CHOP > 61.8) and avoid trend signals in chop
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in both bull/bear: KAMA catches trends, RSI avoids exhaustion, Chop filter prevents whipsaws in ranging markets

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA(10) on 1d timeframe
    def kama(data, period=10):
        # Efficiency Ratio
        change = np.abs(np.diff(data, n=period))
        volatility = np.sum(np.abs(np.diff(data)), axis=0) if len(data) > 1 else 0
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
        # Initialize KAMA
        kama_vals = np.full_like(data, np.nan)
        if len(data) > period:
            kama_vals[period] = data[period]
            for i in range(period+1, len(data)):
                kama_vals[i] = kama_vals[i-1] + sc[i] * (data[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1d = kama(close_1d, 10)
    
    # Calculate RSI(14) on 1d timeframe
    def rsi(data, period=14):
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(data)
        avg_loss = np.zeros_like(data)
        if len(data) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            for i in range(period+1, len(data)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_1d = rsi(close_1d, 14)
    
    # Calculate Choppiness Index(14) on 1d timeframe
    def choppy(data_high, data_low, data_close, period=14):
        # True Range
        tr1 = np.abs(data_high[1:] - data_low[1:])
        tr2 = np.abs(data_high[1:] - data_close[:-1])
        tr3 = np.abs(data_low[1:] - data_close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        max_h = pd.Series(data_high).rolling(window=period, min_periods=period).max().values
        min_l = pd.Series(data_low).rolling(window=period, min_periods=period).min().values
        
        # Chop = 100 * log10(atr / (max_h - min_l)) / log10(period)
        range_hl = max_h - min_l
        chop = np.where((range_hl > 0) & (atr > 0), 
                        100 * np.log10(atr / range_hl) / np.log10(period), 50)
        return chop
    
    chop_1d = choppy(high_1d, low_1d, close_1d, 14)
    
    # Align HTF indicators to 1d timeframe (primary)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price > KAMA AND RSI > 50 AND Chop < 61.8 (trending market)
            if (close[i] > kama_1d_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                chop_1d_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA AND RSI < 50 AND Chop < 61.8 (trending market)
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  chop_1d_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA OR RSI < 40 (momentum loss) OR Chop > 61.8 (chop regime)
            if (close[i] < kama_1d_aligned[i] or 
                rsi_1d_aligned[i] < 40 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA OR RSI > 60 (momentum loss) OR Chop > 61.8 (chop regime)
            if (close[i] > kama_1d_aligned[i] or 
                rsi_1d_aligned[i] > 60 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals