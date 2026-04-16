#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) + 1w EMA trend filter + volume confirmation
# Donchian channels provide clear breakout levels that work in both bull/bear markets.
# 1w EMA filter ensures we trade with the higher timeframe trend.
# Volume confirmation adds conviction to breakout signals.
# ATR-based trailing stop manages risk without look-ahead.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1w data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 12h Donchian Channel (20) ===
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # === 1w EMA Trend Filter (34) ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 12h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 12h ATR (15) for trailing stop ===
    atr_12h = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), np.abs(low_12h - np.roll(close_12h, 1))))
    atr_12h[0] = high_12h[0] - low_12h[0]  # Fix first value
    atr_ma = pd.Series(atr_12h).ewm(span=15, adjust=False, min_periods=15).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume for confirmation
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from high
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from low
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Donchian reversal) ===
        if position == 1:  # Long position
            # Exit when price breaks below lower Donchian band
            if price < lower:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian band
            if price > upper:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_confirm:
                # Long when price breaks above upper Donchian band AND price above 1w EMA (uptrend)
                if price > upper and price > ema_trend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when price breaks below lower Donchian band AND price below 1w EMA (downtrend)
                elif price < lower and price < ema_trend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1wEMA34_VolumeConfirm_ATRTrail"
timeframe = "12h"
leverage = 1.0