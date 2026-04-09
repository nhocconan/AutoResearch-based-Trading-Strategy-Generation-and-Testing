#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ATR-based volume confirmation and ADX regime filter
# - Uses 6h Donchian channel breakouts for entries (structure-based)
# - Volume confirmation: 1d ATR(14) ratio > 1.2 indicating elevated volatility (avoids low-vol breakouts)
# - Regime filter: 1d ADX(14) > 20 to ensure trending market (avoid chop/range)
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in both bull/bear: Donchian adapts to volatility, ADX filter ensures trending conditions,
#   ATR volume confirmation avoids false breakouts in low-volatility environments

name = "6h_1d_donchian_atr_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR ratio: current ATR / 20-period average ATR
    atr_avg_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_avg_20
    atr_ratio = np.where(np.isnan(atr_ratio), 1.0, atr_ratio)
    
    # Calculate 1d ADX(14)
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed ATR, +DM, -DM
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_smoothed = pd.Series(atr_14).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smoothed = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smoothed / atr_14_smoothed)
    minus_di = 100 * (minus_dm_smoothed / atr_14_smoothed)
    plus_di = np.where(np.isnan(plus_di), 0, plus_di)
    minus_di = np.where(np.isnan(minus_di), 0, minus_di)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx = np.where(np.isnan(adx), 0, adx)
    
    # Align 1d indicators to 6h timeframe (completed 1d bar only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h ATR(14) for trailing stop
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.roll(close, 1))
    tr3_6h = np.abs(low - np.roll(close, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or
            np.isnan(atr_ratio_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(atr_6h[i]) or
            atr_6h[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Regime and volatility filters
        volatile_enough = atr_ratio_aligned[i] > 1.2  # ATR > 1.2x average
        trending = adx_aligned[i] > 20  # ADX > 20 indicates trending market
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high
            if low[i] <= highest_since_entry - (2.0 * atr_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low
            if high[i] >= lowest_since_entry + (2.0 * atr_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume and regime confirmation
            # Long: price breaks above Donchian high AND volatile AND trending
            if high[i] >= highest_high_20[i] and volatile_enough and trending:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volatile AND trending
            elif low[i] <= lowest_low_20[i] and volatile_enough and trending:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals