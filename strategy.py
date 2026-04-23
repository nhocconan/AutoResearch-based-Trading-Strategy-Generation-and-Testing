#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume confirmation.
Long when price breaks above 12h Donchian upper band AND price > 1d EMA50 (uptrend) AND volume > 1.5x ATR-scaled average.
Short when price breaks below 12h Donchian lower band AND price < 1d EMA50 (downtrend) AND volume > 1.5x ATR-scaled average.
Exit when price reverts to 12h Donchian midpoint or trend reverses (price crosses 1d EMA50).
Uses 12h timeframe with tight entry conditions (Donchian breakouts are structural) to limit trades.
1d EMA50 provides smooth trend filter. ATR-scaled volume ensures volatility-adjusted confirmation.
Target: 50-150 total trades over 4 years (12-37/year) to stay within proven working range for 12h.
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
    
    # Calculate 12h Donchian channels (20-period) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian channels on 12h (based on previous 20 periods)
    # Upper = max(high over last 20 periods), Lower = min(low over last 20 periods)
    # We'll calculate using rolling window on 12h data
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    midpoint_12h = (upper_12h + lower_12h) / 2.0
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) on 12h for volume scaling
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr2 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with 12h indices
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average scaled by ATR on 12h timeframe
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    # ATR-scaled volume average: higher volatility = higher volume threshold
    vol_threshold_12h = vol_ma_12h * (1 + atr_12h / 100)  # scale by ATR as % of price
    
    # Align HTF indicators to primary timeframe (12h)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    midpoint_12h_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_threshold_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_threshold_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or np.isnan(midpoint_12h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_threshold_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_12h_aligned[i]
        lower_val = lower_12h_aligned[i]
        midpoint_val = midpoint_12h_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_thresh_val = vol_threshold_12h_aligned[i]
        
        # Get current 12h-aligned price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper AND price > 1d EMA50 (uptrend) AND volume > threshold
            if (price > upper_val and price > ema50_val and vol_current > vol_thresh_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower AND price < 1d EMA50 (downtrend) AND volume > threshold
            elif (price < lower_val and price < ema50_val and vol_current > vol_thresh_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 12h Donchian midpoint OR price breaks below 1d EMA50 (trend reversal)
                if price <= midpoint_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 12h Donchian midpoint OR price breaks above 1d EMA50 (trend reversal)
                if price >= midpoint_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA50_ATR_Volume"
timeframe = "12h"
leverage = 1.0