#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
# Long when price breaks above 20-period high AND volume > 1.5x median volume.
# Short when price breaks below 20-period low AND volume > 1.5x median volume.
# Exit when price retraces to 10-period EMA OR ATR-based stoploss (2.5x ATR).
# Uses discrete position size 0.25. Target: 20-50 trades/year to minimize fee drag.
# Donchian channels provide clear structure; volume confirms institutional interest.
# ATR stoploss adapts to volatility, reducing whipsaw in choppy markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 10-period EMA for exit ===
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # === ATR (14-period) for stoploss ===
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # === Volume median (20-period) for confirmation ===
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 10, 14, 20)  # Donchian needs 20, EMA needs 10, ATR needs 14, volume median needs 20
    
    # Track position state and entry price for ATR stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_10[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        atr = atr_14[i]
        vol_median = vol_median_20[i]
        
        # Volume confirmation: current volume > 1.5x median volume
        volume_confirm = volume[i] > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price retraces to 10 EMA OR price drops 2.5x ATR from high
            if (price <= ema_10[i]) or (price < highest_since_entry - 2.5 * atr):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price retraces to 10 EMA OR price rises 2.5x ATR from low
            if (price >= ema_10[i]) or (price > lowest_since_entry + 2.5 * atr):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: break above 20-period high + volume confirmation
            if (price > highest_20[i]) and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            
            # SHORT: break below 20-period low + volume confirmation
            elif (price < lowest_20[i]) and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirm_EMA10Exit_ATRStop2.5_v1"
timeframe = "4h"
leverage = 1.0