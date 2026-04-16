#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels provide clear breakout levels that work in both trending and ranging markets.
# 1d EMA34 establishes the higher timeframe trend bias to avoid counter-trend entries.
# Volume confirmation (>1.5x 20-period average) ensures breakouts have institutional participation.
# ATR-based trailing stop (2.5x ATR) manages risk during strong trends.
# Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA34 trend filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA34 (trend filter) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 4h Donchian(20) channels ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower (20-period)
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # === 4h Volume Confirmation (20-period average) ===
    volume_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # === ATR for trailing stop (4h, 14-period) ===
    atr_period = 14
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    extreme_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_1d_aligned[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        atr_val = atr_aligned[i]
        
        # === TRAILING STOP LOGIC (ATR-based) ===
        if position == 1:  # Long position
            # Update extreme price (highest since entry)
            if price > extreme_price:
                extreme_price = price
            # Trail stop: exit if price drops 2.5*ATR from extreme
            if atr_val > 0 and price < extreme_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update extreme price (lowest since entry)
            if price < extreme_price or extreme_price == 0:
                extreme_price = price
            # Trail stop: exit if price rises 2.5*ATR from extreme
            if atr_val > 0 and price > extreme_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === EXIT LOGIC (Donchian opposite touch) ===
        if position == 1:  # Long position
            # Exit when price touches or crosses below Donchian lower
            if price <= donch_low:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches or crosses above Donchian upper
            if price >= donch_high:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian upper AND above 1d EMA34 AND volume confirmation
            if price > donch_high and price > ema34 and vol_confirm:
                signals[i] = 0.25
                position = 1
                extreme_price = price
                continue
            # Short when: price breaks below Donchian lower AND below 1d EMA34 AND volume confirmation
            elif price < donch_low and price < ema34 and vol_confirm:
                signals[i] = -0.25
                position = -1
                extreme_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeConfirm_ATRTrail"
timeframe = "4h"
leverage = 1.0