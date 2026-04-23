#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above 12h Donchian upper band AND price > 1d EMA50 (uptrend) AND volume > 2.0x average.
Short when price breaks below 12h Donchian lower band AND price < 1d EMA50 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 12h Donchian midpoint OR trend reverses (price crosses 1d EMA50).
Uses 12h timeframe with tight entry conditions (Donchian breakouts are strong momentum signals) to limit trades.
1d EMA50 provides smooth trend filter. Volume spike ensures high-conviction breakouts.
Target: 80-120 trades over 4 years (20-30/year) to stay within proven working range for 12h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian levels (upper, lower, midpoint) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian levels on 12h (based on previous 20-period high/low)
    donch_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_upper + donch_lower) / 2.0
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_12h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_12h, donch_lower)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or np.isnan(donch_mid_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = donch_upper_aligned[i]
        lower_val = donch_lower_aligned[i]
        mid_val = donch_mid_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > upper_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < lower_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 12h Donchian midpoint OR price breaks below 1d EMA50 (trend reversal)
                if price <= mid_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 12h Donchian midpoint OR price breaks above 1d EMA50 (trend reversal)
                if price >= mid_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0