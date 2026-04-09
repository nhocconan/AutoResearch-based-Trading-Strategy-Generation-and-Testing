#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 on 6h
# - Long when Bull Power > 0 AND Bear Power rising (momentum building)
# - Short when Bear Power < 0 AND Bull Power falling (momentum building)
# - Regime filter: 1d ADX > 25 (trending market) to avoid whipsaws in ranges
# - Volume confirmation: 6h volume > 20-period median volume
# - Works in bull/bear: ADX filter ensures we only trade strong trends, Elder Ray captures momentum
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines

name = "6h_1d_elderray_adx_volume_v1"
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
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, period)
    adx_1d = adx  # already smoothed
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA
    bear_power = low_6h - ema_13_6h   # Bear Power = Low - EMA
    
    # 6h volume regime: volume > 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > median_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(ema_13_6h[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power >= 0 (momentum fading) OR ADX < 20 (trend weakening) OR volume drops
            if (bear_power[i] >= 0 or 
                adx_1d_aligned[i] < 20 or 
                not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power <= 0 (momentum fading) OR ADX < 20 (trend weakening) OR volume drops
            if (bull_power[i] <= 0 or 
                adx_1d_aligned[i] < 20 or 
                not volume_regime[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray extremes with volume confirmation and ADX filter
            # Long: Bull Power > 0 AND Bear Power rising (momentum building) AND ADX > 25 AND volume regime
            if (bull_power[i] > 0 and 
                bear_power[i] > bear_power[i-1] and  # Bear Power rising (less negative/more positive)
                adx_1d_aligned[i] > 25 and 
                volume_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power < 0 AND Bull Power falling (momentum building) AND ADX > 25 AND volume regime
            elif (bear_power[i] < 0 and 
                  bull_power[i] < bull_power[i-1] and  # Bull Power falling (less positive/more negative)
                  adx_1d_aligned[i] > 25 and 
                  volume_regime[i]):
                position = -1
                signals[i] = -0.25
    
    return signals