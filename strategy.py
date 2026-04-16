#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h ADX regime filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND ADX > 25 AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Designed to capture trend strength with volume confirmation in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # === 6h Indicators: Elder Ray momentum (change in power) ===
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bull_power_falling = bull_power < np.roll(bull_power, 1)
    bear_power_falling = bear_power < np.roll(bear_power, 1)
    
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
    
    # Directional Movement
    dm_plus = pd.Series(high_12h).diff()
    dm_minus = pd.Series(low_12h).diff().abs()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0.0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    adx_trending = adx_12h_aligned > 25
    
    # === 6h Indicators: Volume Spike ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/EMA)
    warmup = 60
    
    # Track position state and entry price for signal continuity
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
        bull = bull_power[i]
        bear = bear_power[i]
        bull_r = bull_power_rising[i]
        bear_r = bear_power_rising[i]
        bull_f = bull_power_falling[i]
        bear_f = bear_power_falling[i]
        adx_ok = adx_trending[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: reverse signal or loss of conditions ===
        if position == 1:  # Long position
            # Exit if bear power becomes positive (trend weakness) OR ADX weak OR no volume spike
            if bear > 0 or not adx_ok or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit if bull power becomes negative (trend weakness) OR ADX weak OR no volume spike
            if bull < 0 or not adx_ok or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        else:  # Flat - look for entry
            # LONG: Bull Power > 0 AND rising AND trending ADX AND volume spike
            if bull > 0 and bull_r and adx_ok and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power > 0 AND rising AND trending ADX AND volume spike
            elif bear > 0 and bear_r and adx_ok and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0