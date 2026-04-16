#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels provide clear breakout levels that work in both trending and ranging markets.
# Using 1d EMA34 as trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation (>1.5x 20-period average) ensures breakouts have genuine participation.
# ATR-based trailing stop (2.5x ATR) manages risk without being too tight.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.

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
    
    # === 1d EMA34 calculation ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h Donchian(20) calculation ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period high
    upper_band = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower_band = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # === 12h Volume Confirmation (20-period average) ===
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 12h ATR for trailing stop (14-period) ===
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]  # First period has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    extreme_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            extreme_price = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_aligned[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        vol_ma = vol_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol_confirm = volume[i] > vol_ma * 1.5  # 1.5x average volume
        
        # Update trailing stop extreme price
        if position == 1:  # Long position
            if price > extreme_price:
                extreme_price = price
        elif position == -1:  # Short position
            if extreme_price == 0 or price < extreme_price:
                extreme_price = price
        
        # === TRAILING STOP LOGIC ===
        if position == 1 and atr_val > 0:
            # Exit long if price drops 2.5*ATR from extreme high
            if price < extreme_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        elif position == -1 and atr_val > 0:
            # Exit short if price rises 2.5*ATR from extreme low
            if price > extreme_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above upper band AND above 1d EMA34 AND volume confirmation
            if price > upper and price > ema34 and vol_confirm:
                signals[i] = 0.25
                position = 1
                extreme_price = price
                continue
            # Short when: price breaks below lower band AND below 1d EMA34 AND volume confirmation
            elif price < lower and price < ema34 and vol_confirm:
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

name = "12h_Donchian20_1dEMA34_VolumeConfirm_ATRTrail"
timeframe = "12h"
leverage = 1.0