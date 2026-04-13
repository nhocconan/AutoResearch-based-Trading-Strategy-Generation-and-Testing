#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Follow_Strategy
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, reducing false signals in chop.
Combined with 1-week trend filter and volume confirmation, this captures strong trends while avoiding whipsaws.
Works in both bull (adapts to strong uptrends) and bear (adapts to strong downtrends) markets.
Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA on daily prices
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle the array shapes correctly
    change_padded = np.concatenate([[np.nan]*10, change])
    volatility_padded = np.concatenate([[np.nan]*1, volatility])
    
    # Calculate ER and volatility arrays properly
    er = np.full_like(close, np.nan)
    vol_sum = np.full_like(close, np.nan)
    
    for i in range(10, len(close)):
        if i >= 10:
            ch = np.abs(close[i] - close[i-10])
            vol = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if vol != 0:
                er[i] = ch / vol
            else:
                er[i] = 1.0
            vol_sum[i] = vol
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(kama[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above KAMA (uptrend signal)
        # 2. Price above weekly EMA20 (1w trend filter)
        # 3. Volume expansion
        price_above_kama = close[i] > kama[i]
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        long_condition = price_above_kama and price_above_weekly_ema and volume_expansion[i]
        
        # Short conditions:
        # 1. Price below KAMA (downtrend signal)
        # 2. Price below weekly EMA20 (1w trend filter)
        # 3. Volume expansion
        price_below_kama = close[i] < kama[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        short_condition = price_below_kama and price_below_weekly_ema and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_KAMA_Trend_Follow_Strategy"
timeframe = "1d"
leverage = 1.0