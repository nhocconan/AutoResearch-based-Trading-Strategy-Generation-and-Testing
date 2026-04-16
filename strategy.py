#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume confirmation.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 12h ADX > 25 AND volume > 1.5x 20-period average.
# Exit when Elder Ray signals reverse (Bull Power < 0 for long exit, Bear Power > 0 for short exit) or opposite Elder Ray signal.
# Uses discrete position size 0.25. Designed to capture trending moves with strength confirmation in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # === 12h Indicators: ADX(14) for regime filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = pd.Series(low_12h).diff().abs()
    tr3 = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = pd.Series(high_12h).diff()
    dm_minus = pd.Series(low_12h).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0.0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0.0)
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    adx_trending = adx_12h_aligned > 25
    
    # === 12h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/EMA)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        adx_ok = adx_trending[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power becomes negative (weakening uptrend)
            if bull_val < 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power becomes positive (weakening downtrend)
            if bear_val > 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power rising (less negative) AND ADX trending AND volume spike
            if bull_val > 0 and bear_val > bear_power[i-1] and adx_ok and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND Bull Power falling (less positive) AND ADX trending AND volume spike
            elif bear_val < 0 and bull_val < bull_power[i-1] and adx_ok and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0