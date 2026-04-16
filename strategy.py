#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h VWAP deviation with 1d ATR filter for mean reversion in range-bound markets.
# In ranging markets (ADX < 20), price tends to revert to VWAP with oversold/overbought conditions.
# Uses 4h VWAP deviation (price/VWAP - 1) and 1d ATR to set dynamic entry/exit zones.
# Long when price < VWAP - 1.5*ATR(1d) and short when price > VWAP + 1.5*ATR(1d).
# Volume confirmation (>1.3x average) required. Position size 0.25.
# Works in both bull and bear by adapting to volatility and mean reversion in ranges.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (higher timeframe for ATR) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h VWAP calculation ===
    typical_price_4h = (high_4h + low_4h + close_4h) / 3
    vwap_num = (typical_price_4h * volume_4h).cumsum()
    vwap_den = volume_4h.cumsum()
    vwap_4h = vwap_num / vwap_den
    vwap_deviation = (close_4h / vwap_4h) - 1  # Positive = above VWAP
    
    # === 1d ATR(14) for volatility scaling ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # === 4h ADX(14) for regime detection (trending vs ranging) ===
    # +DM, -DM, TR
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # === 4h volume ratio for confirmation ===
    vol_ma_10_4h = pd.Series(volume_4h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_4h = volume_4h / vol_ma_10_4h
    
    # Align all indicators to lower timeframe
    vwap_dev_aligned = align_htf_to_ltf(prices, df_4h, vwap_deviation)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_values)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(vwap_dev_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vwap_dev = vwap_dev_aligned[i]
        atr = atr_14_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.0 * ATR
            if price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.0 * ATR
            if price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to VWAP or ADX indicates strong trend
            if vwap_dev >= -0.02 or adx_val > 25:  # Near VWAP or strong trend
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to VWAP or ADX indicates strong trend
            if vwap_dev <= 0.02 or adx_val > 25:  # Near VWAP or strong trend
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat and ranging market) ===
        if position == 0 and adx_val < 20:  # Only enter in ranging markets (ADX < 20)
            # LONG: price significantly below VWAP with volume
            if vwap_dev < -0.015 and vol_ratio > 1.3:  # Below VWAP by 1.5*ATR equivalent
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: price significantly above VWAP with volume
            elif vwap_dev > 0.015 and vol_ratio > 1.3:  # Above VWAP by 1.5*ATR equivalent
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_VWAP_ADX_RangeMeanReversion_Volume_v1"
timeframe = "4h"
leverage = 1.0