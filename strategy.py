#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Bollinger Bands (20,2) ===
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # === 1d Bollinger Band Width Percentile (50) ===
    df_1d = get_htf_data(prices, '1d')
    bb_mid_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    bb_std_1d = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    bb_width_1d = (bb_mid_1d + 2 * bb_std_1d) - (bb_mid_1d - 2 * bb_std_1d)
    bb_width_pct = pd.Series(bb_width_1d).rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Align 1d BB width percentile to 6h
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    
    # === 1d ADX (14) for Trend Strength ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 60  # Need BB20, BBwidth50, ADX14
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(bb_width_pct_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bb_width_pct_val = bb_width_pct_aligned[i]
        adx_val = adx_aligned[i]
        
        # === EXIT LOGIC: Exit when price returns to BB middle ===
        if position == 1:  # Long position
            # Exit when price crosses back below BB middle
            if price < bb_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above BB middle
            if price > bb_mid[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price touches lower BB in low volatility (squeeze) AND weak trend
            if price <= bb_lower[i] and bb_width_pct_val < 0.3 and adx_val < 25:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price touches upper BB in low volatility (squeeze) AND weak trend
            elif price >= bb_upper[i] and bb_width_pct_val < 0.3 and adx_val < 25:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_BollingerSqueeze_1d_ADX_Filter"
timeframe = "6h"
leverage = 1.0