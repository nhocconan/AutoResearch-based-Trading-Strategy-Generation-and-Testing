#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h volume filter and 1d trend filter (ADX > 25).
# Long when fast EMA crosses above slow EMA AND 4h volume > 1.8x 20-period average AND 1d ADX > 25.
# Short when fast EMA crosses below slow EMA AND 4h volume > 1.8x 20-period average AND 1d ADX > 25.
# Exit when fast EMA crosses back through slow EMA.
# Uses discrete position size 0.20. Designed to capture momentum in trending markets with volume confirmation.
# Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: EMA Crossover (fast=9, slow=21) ===
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 4h Indicators: Volume Spike (volume > 1.8x 20-period average) ===
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    volume_spike = volume > (1.8 * vol_ma_4h_aligned)
    
    # === 1d Indicators: ADX > 25 (trending market filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * (dm_plus_smooth / atr_smooth)
    di_minus = 100 * (dm_minus_smooth / atr_smooth)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trending = adx_aligned > 25
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/ATR/EMA)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(volume_spike[i]) or
            np.isnan(trending[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_trending = trending[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if fast EMA crosses below slow EMA
            if ema_fast[i] < ema_slow[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if fast EMA crosses above slow EMA
            if ema_fast[i] > ema_slow[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Fast EMA crosses above slow EMA AND volume spike AND trending market
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and vol_spike and is_trending:
                signals[i] = 0.20
                position = 1
            
            # SHORT: Fast EMA crosses below slow EMA AND volume spike AND trending market
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and vol_spike and is_trending:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA9_21_Crossover_Volume4h_ADX1d_V1"
timeframe = "1h"
leverage = 1.0