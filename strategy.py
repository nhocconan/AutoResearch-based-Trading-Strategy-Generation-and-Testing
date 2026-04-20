#!/usr/bin/env python3
"""
12h_KAMA_Direction_Plus_RSI_With_Chop_Filter
Hypothesis: KAMA on 12h identifies trend direction, RSI(2) on 12h for mean-reversion entries, and Choppiness Index on 1d filters regime (chop > 61.8 = range for mean reversion). Works in both bull and bear by fading extremes in ranging markets while avoiding strong trends.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "12h_KAMA_Direction_Plus_RSI_With_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if hasattr(np, 'sum') else np.abs(np.diff(close, n=1)).sum()
    # Handle array case
    if len(change.shape) == 0:
        er = 0 if volatility == 0 else change / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = np.zeros_like(close)
        kama[0] = close
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        return kama
    else:
        # Vectorized version for arrays
        change = np.abs(np.subtract(close, np.roll(close, er_period)))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
        er = np.where(volatility == 0, 0, np.divide(change, volatility))
        sc = np.power(np.add(np.multiply(er, (fast_sc - slow_sc)), slow_sc), 2)
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

def calculate_rsi(close, period=2):
    """RSI calculation"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index"""
    atr = np.zeros_like(close)
    tr = np.zeros_like(close)
    
    for i in range(len(close)):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # ATR calculation
    atr[:period-1] = np.nan
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Choppiness
    sum_tr = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            sum_tr[i] = np.nan
        else:
            sum_tr[i] = np.sum(tr[i-period+1:i+1])
    
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            max_high[i] = np.nan
            min_low[i] = np.nan
        else:
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
    
    chop = np.where(
        (max_high - min_low) == 0,
        50,
        100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(period)
    )
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for Choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate indicators on 12h
    kama_12h = calculate_kama(close_12h, 10, 2, 30)
    rsi_12h = calculate_rsi(close_12h, 2)
    
    # Calculate Choppiness on 1d
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Align to 12h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(close[i])):
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
        
        # Determine trend direction from KAMA
        trend_up = close_12h[i // 12] > kama_12h[i // 12] if i >= 12 else False
        trend_down = close_12h[i // 12] < kama_12h[i // 12] if i >= 12 else False
        
        if position == 0:
            # Only trade in ranging markets (Choppiness > 61.8)
            if chop_1d_aligned[i] > 61.8:
                # Mean reversion: buy when RSI < 30 (oversold)
                if rsi_12h_aligned[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Sell when RSI > 70 (overbought)
                elif rsi_12h_aligned[i] > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (mean reversion complete) or chop < 38.2 (trending)
            if rsi_12h_aligned[i] > 50 or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (mean reversion complete) or chop < 38.2 (trending)
            if rsi_12h_aligned[i] < 50 or chop_1d_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals