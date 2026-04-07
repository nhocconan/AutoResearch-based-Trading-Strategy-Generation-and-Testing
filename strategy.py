#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray + ADX filter + 1d weekly pivot direction
# Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and price > 1d weekly pivot (R1)
# Short when Bear Power < 0, Bull Power < 0, ADX > 25 (trending), and price < 1d weekly pivot (S1)
# Exit when Elder Power signs flip or ADX < 20 (range)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses Elder Ray (Bull/Bear Power) from 6h, ADX from 6h, weekly pivot from 1d
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_elder_ray_adx_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h data for Elder Ray and ADX
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False).mean().values
    bull_power = high_6h - ema13
    bear_power = low_6h - ema13
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                          np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        return adx
    
    adx = calculate_adx(high_6h, low_6h, close_6h, 14)
    
    # Align 6h indicators to lower timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_6h, adx)
    
    # 1d data for weekly pivot points (using prior week's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from prior week's daily data
    # Using prior week's high, low, close (5-day lookback)
    def calculate_weekly_pivot(high, low, close):
        # Need at least 5 days for weekly
        if len(high) < 5:
            return np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # Rolling weekly high, low, close (prior week)
        weekly_high = pd.Series(high).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        weekly_low = pd.Series(low).rolling(window=5, min_periods=5).min().shift(1)    # Prior week
        weekly_close = pd.Series(close).rolling(window=5, min_periods=5).last().shift(1) # Prior week close
        
        # Pivot point
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Support and resistance levels
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pivot)
        
        return r1.values, s1.values
    
    r1, s1 = calculate_weekly_pivot(high_1d, low_1d, close_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Power flip or ADX < 20 (range) or price breaks below S1
            elif (bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or 
                  adx_aligned[i] < 20 or close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Power flip or ADX < 20 (range) or price breaks above R1
            elif (bull_power_aligned[i] >= 0 or bear_power_aligned[i] <= 0 or 
                  adx_aligned[i] < 20 or close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Elder Ray alignment, ADX filter, and pivot filter
            # Long: Bull Power > 0, Bear Power < 0, ADX > 25, price > R1
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                adx_aligned[i] > 25 and close[i] > r1_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bull Power < 0, Bear Power > 0, ADX > 25, price < S1
            elif (bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and 
                  adx_aligned[i] > 25 and close[i] < s1_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals