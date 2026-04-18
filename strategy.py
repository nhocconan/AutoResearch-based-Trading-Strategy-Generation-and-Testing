#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation
Hypothesis: Price breaking above/below weekly Donchian Channel (20-period high/low) 
with volume confirmation (volume > 1.5x daily average) indicates strong momentum. 
Weekly timeframe filters daily noise, reducing false breakouts. Works in bull/bear 
by capturing strong trends. Target: 15-25 trades/year to minimize fee drag.
"""

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
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian Channel (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (waits for weekly bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Volume confirmation: volume > 1.5x 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators (max of 20 weekly + 20 daily)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Price breaks above weekly Donchian high with volume confirmation = long
            if price > upper and vol_conf:
                signals[i] = 0.25
                position = 1
            # Price breaks below weekly Donchian low with volume confirmation = short
            elif price < lower and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to weekly Donchian low (mean reversion) or trend weakens
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly Donchian high (mean reversion) or trend weakens
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume"
timeframe = "1d"
leverage = 1.0