#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with weekly trend filter
# Long when Bull Power > 0 (price > EMA13) and weekly close > weekly EMA40 (uptrend)
# Short when Bear Power < 0 (price < EMA13) and weekly close < weekly EMA40 (downtrend)
# Exit when Elder Power reverses sign or weekly trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25
# Uses weekly EMA40 for trend filter and Elder Power for entry/exit
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_elder_ray_weekly_ema40_v1"
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
    
    # 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_6h - ema13
    # Bear Power = Low - EMA13
    bear_power = low_6h - ema13
    
    # Align Elder Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Weekly data for EMA40 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 40:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    ema40_weekly = pd.Series(close_weekly).ewm(span=40, adjust=False).mean().values
    ema40_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema40_weekly)
    
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
            np.isnan(ema40_weekly_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Power turns negative OR weekly trend turns down
            elif bear_power_aligned[i] >= 0 or close[i] < ema40_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Elder Power turns positive OR weekly trend turns up
            elif bull_power_aligned[i] <= 0 or close[i] > ema40_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Elder Power alignment and weekly trend
            # Long: Bull Power positive AND price above weekly EMA40 (uptrend)
            if bull_power_aligned[i] > 0 and close[i] > ema40_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power negative AND price below weekly EMA40 (downtrend)
            elif bear_power_aligned[i] < 0 and close[i] < ema40_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals