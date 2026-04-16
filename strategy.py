#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and 4h volume confirmation.
# Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 1.8x 20-bar average.
# Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 1.8x 20-bar average.
# Exit when price reverts to Donchian midpoint (mean reversion exit).
# Uses discrete position size 0.25. Donchian channels provide clear breakout levels with built-in volatility adjustment.
# 1d EMA50 ensures we trade with higher timeframe trend. Volume confirms breakout strength.
# Target: 80-160 trades over 4 years (20-40/year) to avoid fee drag while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper band: highest high over 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    middle_band = (highest_high + lowest_low) / 2.0
    
    # === 4h Indicators: Volume MA (20) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data once before loop for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        upper = highest_high[i]
        lower = lowest_low[i]
        middle = middle_band[i]
        vol_ma_val = vol_ma_20[i]
        ema50_val = ema50_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: volume > 1.8x 20-period average
        vol_filter = vol > 1.8 * vol_ma_val if vol_ma_val > 0 else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverts to middle band (mean reversion)
            if price <= middle:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverts to middle band (mean reversion)
            if price >= middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band AND price > 1d EMA50 AND volume confirmation
            if price > upper and price > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower band AND price < 1d EMA50 AND volume confirmation
            elif price < lower and price < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0