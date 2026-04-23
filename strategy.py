#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band (20-period high) AND price > 1d EMA50 (uptrend) AND volume > 2.0x average.
Short when price breaks below Donchian lower band (20-period low) AND price < 1d EMA50 (downtrend) AND volume > 2.0x average.
Exit when price reverts to Donchian midpoint OR trend reverses (price crosses 1d EMA50).
Designed for low trade frequency (~10-20/year) to capture strong breakouts in trending markets while avoiding false signals.
Uses Donchian channels for structure, EMA50 for trend, volume for confirmation - proven combo for BTC/ETH in both bull/bear.
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
    
    # Load 1d data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 for 1d trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma + low_ma) / 2
    
    # Align HTF indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_band = high_ma[i]
        lower_band = low_ma[i]
        mid_band = donchian_mid[i]
        ema50_val = ema50_aligned[i]
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper band AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > upper_band and price > ema50_val and vol_current > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < lower_band and price < ema50_val and vol_current > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to midpoint OR price breaks below 1d EMA50 (trend reversal)
                if price <= mid_band or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to midpoint OR price breaks above 1d EMA50 (trend reversal)
                if price >= mid_band or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_Volume_Breakout"
timeframe = "12h"
leverage = 1.0