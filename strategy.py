#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and 1d volume confirmation.
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low.
# Long when Bull Power > 0, ADX > 25 (trending), and daily volume > 1.5x its 20-period median.
# Short when Bear Power > 0, ADX > 25 (trending), and daily volume > 1.5x its 20-period median.
# Exit when power reverses sign or ADX < 20 (range market).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Combines momentum (Elder Ray) with trend strength (ADX) and volume confirmation for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once before loop for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # === 6h Indicators: Elder Ray (Bull/Bear Power) ===
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA13 of close
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power_6h = high_6h - ema_13_6h
    # Bear Power = EMA13 - Low
    bear_power_6h = ema_13_6h - low_6h
    
    # Get daily data for ADX and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Indicators: ADX (14) and Volume ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # first value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / atr_1d
    di_minus = 100 * dm_minus_smoothed / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume median
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 30)  # 6h EMA13, daily ADX, daily volume median
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        adx = adx_aligned[i]
        vol_median = vol_median_aligned[i]
        daily_volume = vol_1d_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when Bull Power <= 0 (momentum fading) OR ADX < 20 (range market)
            if bull_power <= 0 or adx < 20:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when Bear Power <= 0 (momentum fading) OR ADX < 20 (range market)
            if bear_power <= 0 or adx < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume confirmation: current daily volume > 1.5x its 20-period median
            volume_confirm = daily_volume > (vol_median * 1.5)
            # Trend filter: ADX > 25 (strong trend)
            trending = adx > 25
            
            # LONG CONDITIONS
            # Bull Power > 0 AND trending AND volume confirmation
            if bull_power > 0 and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Bear Power > 0 AND trending AND volume confirmation
            elif bear_power > 0 and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_ElderRay_1dADX25_VolumeConfirm1.5x_v1"
timeframe = "6h"
leverage = 1.0