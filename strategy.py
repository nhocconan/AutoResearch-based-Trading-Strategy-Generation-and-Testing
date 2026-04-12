#!/usr/bin/env python3
"""
6h_1d_PowerOfThree_V1
Hypothesis: Uses Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
Long when Bull Power > 0 and Bear Power < 0 in uptrend; short when Bear Power > 0 and Bull Power < 0 in downtrend.
Designed for low trade frequency by requiring strong directional power and trend alignment.
Works in bull via long strength, in bear via short strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_PowerOfThree_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for power calculation
    close_s = pd.Series(close_1d)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13
    bear_power = ema13 - low_1d
    
    # Daily SMA50 for trend filter
    sma50 = close_s.rolling(window=50, min_periods=50).mean().values
    
    # Align to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50)
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(sma50_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_avg[i]
        
        # Trend filter: price above/below SMA50
        price_vs_sma = close[i] > sma50_aligned[i]
        
        # Power conditions
        bull_strong = bull_power_aligned[i] > 0
        bear_strong = bear_power_aligned[i] > 0
        
        # Entry conditions
        long_setup = bull_strong and not bear_strong and price_vs_sma and vol_confirm
        short_setup = bear_strong and not bull_strong and not price_vs_sma and vol_confirm
        
        # Exit when power reverses or trend fails
        exit_long = not bull_strong or not price_vs_sma
        exit_short = not bear_strong or price_vs_sma
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals