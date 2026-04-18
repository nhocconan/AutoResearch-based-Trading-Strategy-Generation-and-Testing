#!/usr/bin/env python3
"""
4h Ehlers Fisher Transform with 12h Trend Filter and Volume Confirmation
Hypothesis: Ehlers Fisher Transform (EFT) detects extreme price reversals with minimal lag.
In trending markets (12h EMA34), EFT crossovers signal continuation; in ranging markets,
they signal reversals. Volume confirms breakout strength. Designed for low trade frequency
(~25-40/year) to avoid fee drag while capturing major moves in both bull and bear markets.
"""

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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Ehlers Fisher Transform (length=10)
    # Price = (High + Low) / 2
    hl2 = (high + low) / 2
    # Normalize price to [-1, 1] using 10-period min/max
    min_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    max_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    # Normalized price: -1 to 1
    price_norm = 2 * ((hl2 - min_low) / range_hl) - 1
    # Clamp to avoid math domain errors
    price_norm = np.clip(price_norm, -0.999, 0.999)
    # Fisher Transform: 0.5 * ln((1 + x) / (1 - x))
    fish = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
    # Smoothed Fisher (3-period EMA)
    fish_smooth = pd.Series(fish).ewm(span=3, adjust=False, min_periods=3).mean().values
    # Trigger line (1-period delay)
    fish_trigger = np.roll(fish_smooth, 1)
    fish_trigger[0] = 0  # No trigger for first bar
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(fish_smooth[i]) or np.isnan(fish_trigger[i]):
            signals[i] = 0.0
            continue
        
        trend = ema34_12h_aligned[i]
        vol_ok = vol_confirm[i]
        fish_val = fish_smooth[i]
        fish_trig = fish_trigger[i]
        
        if position == 0:
            # Enter long: Fisher crosses above trigger with volume + uptrend
            if fish_val > fish_trig and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: Fisher crosses below trigger with volume + downtrend
            elif fish_val < fish_trig and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Fisher crosses below trigger or trend turns down
            if fish_val < fish_trig or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Fisher crosses above trigger or trend turns up
            if fish_val > fish_trig or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Ehlers_Fisher_Transform_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0