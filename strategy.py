#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Donchian channels provide clear breakout levels that work in both trending and ranging markets.
# 1d ADX > 25 filters for strong trends, avoiding whipsaws in choppy markets.
# Volume confirmation (>1.5x 20-period average) ensures breakouts have institutional participation.
# ATR-based trailing stop (2.5x ATR) manages risk while allowing trends to develop.
# Target: 100-180 total trades over 4 years (25-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for ADX calculation (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === ADX Calculation (1d) ===
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(series, period):
        result = np.zeros_like(series)
        if len(series) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(series[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr.values, period)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 4h Donchian Channel (20-period) ===
    donchian_period = 20
    dc_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    dc_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 4h ATR for Trailing Stop (14-period) ===
    atr_period = 14
    tr_4h = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr_4h[0] = high[0] - low[0]  # First TR
    atr_4h = np.zeros_like(tr_4h)
    if len(tr_4h) >= atr_period:
        atr_4h[atr_period-1] = np.mean(tr_4h[:atr_period])
        for i in range(atr_period, len(tr_4h)):
            atr_4h[i] = (atr_4h[i-1] * (atr_period-1) + tr_4h[i]) / atr_period
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = max(donchian_period, 20, atr_period) + 5
    
    # Track position and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    extreme_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(dc_high[i]) or
            np.isnan(dc_low[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            position = 0
            extreme_price = 0.0
            continue
        
        price = close[i]
        adx = adx_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_4h[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update extreme price (highest since entry)
            if price > extreme_price:
                extreme_price = price
            # Trail stop: exit if price drops 2.5*ATR from extreme
            if atr > 0 and price < extreme_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update extreme price (lowest since entry)
            if price < extreme_price or extreme_price == 0:
                extreme_price = price
            # Trail stop: exit if price rises 2.5*ATR from extreme
            if atr > 0 and price > extreme_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === EXIT LOGIC (ADX trend weakness) ===
        if position != 0:
            # Exit when ADX falls below 20 (trend weakening)
            if adx < 20.0:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            vol_confirm = volume[i] > vol_ma * 1.5  # 1.5x average volume
            
            # Long when: Price breaks above Donchian high AND ADX > 25 AND volume confirmation
            if price > dc_high[i] and adx > 25.0 and vol_confirm:
                signals[i] = 0.25
                position = 1
                extreme_price = price
                continue
            # Short when: Price breaks below Donchian low AND ADX > 25 AND volume confirmation
            elif price < dc_low[i] and adx > 25.0 and vol_confirm:
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

name = "4h_Donchian20_1dADXtrend_VolumeConfirm_ATRTrail"
timeframe = "4h"
leverage = 1.0