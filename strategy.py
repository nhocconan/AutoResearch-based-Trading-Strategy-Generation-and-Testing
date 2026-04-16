# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R combined with 1d EMA trend filter and volume confirmation.
# Williams %R(14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold.
# In trending markets (price > 1d EMA50): short at overbought, long at oversold with continuation.
# In ranging markets (price near 1d EMA50): mean reversion at extreme %R levels.
# Volume confirmation (>1.3x average) filters weak signals. Position size 0.25 for risk control.
# Designed to work in bull (trend continuations) and bear (mean reversion in ranges).

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
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for EMA trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 12h Williams %R(14) ===
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # === 1d EMA(50) for trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        ema50 = ema_50_aligned[i]
        vol_ratio = vol_ratio_12h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Dynamic stop loss based on volatility
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Dynamic stop loss based on volatility
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Williams %R reaches overbought or trend changes
            if wr > -20 or price < ema50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R reaches oversold or trend changes
            if wr < -80 or price > ema50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine market regime based on price vs EMA
            if price > ema50 * 1.02:  # Strong uptrend (>2% above EMA)
                # Look for oversold pullback to enter long
                if wr < -80 and vol_ratio > 1.3:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
            elif price < ema50 * 0.98:  # Strong downtrend (>2% below EMA)
                # Look for overbought rally to enter short
                if wr > -20 and vol_ratio > 1.3:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
            else:  # Ranging market (near EMA)
                # Mean reversion at extreme Williams %R levels
                if wr < -85 and vol_ratio > 1.3:  # Deep oversold
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                elif wr > -15 and vol_ratio > 1.3:  # Deep overbought
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

name = "12h_WilliamsR_EMA50_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0