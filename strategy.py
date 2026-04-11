#!/usr/bin/env python3
"""
6h_1d_elder_ray_power_zone_v1
Strategy: 6h Elder Ray (Bull/Bear Power) with 1d zone filter and volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Elder Ray measures bull/bear power via EMA13. In strong trends, power persists; in reversals, power diverges from price. We enter when 6th power aligns with 1d trend zone (above/below EMA50) and volume confirms. Exit when power crosses zero or reverses. Designed to work in both bull (buy strength) and bear (sell weakness) markets with controlled frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_elder_ray_power_zone_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # 1d EMA50 for trend zone
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_avg  # Above average volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Determine 1d trend zone
        above_1d_zone = price_close > ema50_1d_aligned[i]   # Bullish zone
        below_1d_zone = price_close < ema50_1d_aligned[i]   # Bearish zone
        
        # Elder Ray signals with zone alignment
        long_signal = (bull_power[i] > 0) and above_1d_zone and vol_filter[i]
        short_signal = (bear_power[i] < 0) and below_1d_zone and vol_filter[i]
        
        # Exit when power crosses zero or reverses against zone
        exit_long = position == 1 and (bull_power[i] <= 0 or not above_1d_zone)
        exit_short = position == -1 and (bear_power[i] >= 0 or not below_1d_zone)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals