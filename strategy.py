#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 6h Williams %R with 1d ADX regime filter and volume confirmation.
# Long when Williams %R crosses above -80 (oversold bounce) in bullish regime (1d ADX > 25) with volume spike.
# Short when Williams %R crosses below -20 (overbought rejection) in bearish regime (1d ADX > 25) with volume spike.
# Uses 1d ADX to filter regimes: only trade mean reversion in strong trends (ADX > 25) to avoid whipsaws in ranging markets.
# Williams %R(14) measures momentum extremes; ADX > 25 ensures we trade with the trend, not against it.
# Discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for Williams %R and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # === 6h Indicators: Williams %R(14) and volume median ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    vol_6h = df_6h['volume'].values
    
    # Calculate 6h Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume median (20-period)
    vol_median_20 = pd.Series(vol_6h).rolling(window=20, min_periods=20).median().values
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: ADX(14) for regime filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(close_1d).shift(1)
    tr3 = pd.Series(low_1d) - pd.Series(close_1d).shift(1)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff().values
    down_move = -pd.Series(low_1d).diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    vol_median_aligned = align_htf_to_ltf(prices, df_6h, vol_median_20)
    vol_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_6h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 14)  # volume median(20), Williams %R(14), ADX(14)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_6h_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        wr = williams_r_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_6h = vol_6h_aligned[i]
        adx_val = adx_aligned[i]
        
        # Get previous Williams %R for crossover detection
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        
        # Volume spike filter: current 6h volume > 1.3x median volume
        volume_spike = vol_6h > (vol_median * 1.3)
        
        # Regime filter: only trade when 1d ADX > 25 (strong trend)
        strong_trend = adx_val > 25
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when Williams %R crosses above -20 (overbought)
            if wr_prev <= -20 and wr > -20:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when Williams %R crosses below -80 (oversold)
            if wr_prev >= -80 and wr < -80:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0 and strong_trend and volume_spike:
            # LONG: Williams %R crosses above -80 (oversold bounce) in strong trend
            if wr_prev <= -80 and wr > -80:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R crosses below -20 (overbought rejection) in strong trend
            elif wr_prev >= -20 and wr < -20:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_WilliamsR_ADX25_VolumeSpike1.3x_v1"
timeframe = "6h"
leverage = 1.0