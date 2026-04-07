#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Supertrend + 12h ADX + Volume Confirmation
# Hypothesis: Combines trend following (Supertrend) with trend strength filter (ADX) and volume
# to avoid whipsaws. Works in bull via trend following, in bear via short signals when ADX > 25.
# Target: 20-40 trades/year to minimize fee drag.
name = "6h_supertrend_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(x[1:period])
        # Subsequent values
        for i in range(period, len(x)):
            if not np.isnan(result[i-1]) and not np.isnan(x[i]):
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, df_12h, adx)
    
    # Supertrend on 6h data
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for Supertrend
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h = np.concatenate([[np.nan], tr_6h])
    
    atr_6h = wilders_smoothing(tr_6h, atr_period)
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr_6h)
    lower_band = hl2 - (multiplier * atr_6h)
    
    supertrend = np.full(n, np.nan)
    dir_ = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    if not np.isnan(atr_6h[atr_period]):
        supertrend[atr_period] = lower_band[atr_period]
        dir_[atr_period] = 1
    
    for i in range(atr_period + 1, n):
        if np.isnan(atr_6h[i]) or np.isnan(atr_6h[i-1]):
            supertrend[i] = supertrend[i-1]
            dir_[i] = dir_[i-1]
            continue
            
        if close[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
            if dir_[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if dir_[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if dir_[i] == 1 else upper_band[i]
    
    # Volume confirmation: 6h volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(adx_6h[i]) or np.isnan(supertrend[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: Supertrend turns bearish or ADX weakens (< 20)
            if dir_[i] == -1 or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Supertrend turns bullish or ADX weakens (< 20)
            if dir_[i] == 1 or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Supertrend bullish, ADX strong (> 25), volume confirmation
            if dir_[i] == 1 and adx_6h[i] > 25 and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: Supertrend bearish, ADX strong (> 25), volume confirmation
            elif dir_[i] == -1 and adx_6h[i] > 25 and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals