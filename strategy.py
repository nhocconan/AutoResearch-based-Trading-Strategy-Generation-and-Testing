#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ADX trend filter and volume confirmation
# Works in bull/bear by using ADX to filter only trending markets (ADX>25) and avoiding range-bound chop
# Targets 15-25 trades/year to minimize fee drag while capturing major trends
# Uses 12h timeframe for lower trade frequency vs 4h, reducing churn

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
    
    # 12h Donchian channels (20 periods) for breakout signals
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_20_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_20_12h)
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX calculation (14 periods) for trend strength
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Volume confirmation (12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Position tracking
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is invalid
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper_12h = donchian_upper_12h[i]
        lower_12h = donchian_lower_12h[i]
        adx_val = adx_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Exit conditions
        if position == 1:  # Long position
            if price < lower_12h:  # Exit on Donchian lower break
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            if price > upper_12h:  # Exit on Donchian upper break
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # Require strong trend (ADX > 25) and volume confirmation (vol_ratio > 1.5)
            if adx_val > 25 and vol_ratio_val > 1.5:
                # LONG: Price breaks above Donchian upper
                if price > upper_12h:
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Price breaks below Donchian lower
                elif price < lower_12h:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_ADX25_Volume"
timeframe = "12h"
leverage = 1.0