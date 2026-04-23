#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 1d Donchian upper band AND price > 1w EMA50 (uptrend) AND volume > 2.0x average.
Short when price breaks below 1d Donchian lower band AND price < 1w EMA50 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 1d Donchian middle band (20-period mean) or trend reverses (price crosses 1w EMA50).
Uses 1d timeframe with tight entry conditions (Donchian breakouts are strong momentum signals) to limit trades.
1w EMA50 provides smooth trend filter. Volume spike ensures high-conviction breakouts.
Target: 30-100 total trades over 4 years (7-25/year) to stay within proven working range for 1d timeframe.
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
    
    # Calculate 1d Donchian channels (20-period) - ONCE before loop
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0  # middle band
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_band = high_20[i]
        lower_band = low_20[i]
        mid_band = mid_20[i]
        ema50_val = ema50_1w_aligned[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > upper_band and price > ema50_val and vol_current > 2.0 * vol_ma_primary):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < lower_band and price < ema50_val and vol_current > 2.0 * vol_ma_primary):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 1d Donchian middle band OR price breaks below 1w EMA50 (trend reversal)
                if price <= mid_band or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 1d Donchian middle band OR price breaks above 1w EMA50 (trend reversal)
                if price >= mid_band or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Volume_Breakout"
timeframe = "1d"
leverage = 1.0