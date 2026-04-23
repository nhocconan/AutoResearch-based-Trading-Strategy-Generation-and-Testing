#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX Regime Filter + Volume Spike Confirmation.
Long when: Bull Power > 0 (close > EMA13), 1d ADX > 25 (trending), and 6h volume > 1.5x 20-period MA.
Short when: Bear Power < 0 (close < EMA13), 1d ADX > 25 (trending), and 6h volume > 1.5x 20-period MA.
Exit when: Bull/Bear Power crosses zero OR 1d ADX < 20 (regime shift to ranging) OR volume normalizes.
Uses 1d HTF for ADX regime filter to avoid false breakouts in choppy markets, Elder Power for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray avoids whipsaws by measuring power relative to EMA, ADX filters non-trending regimes.
"""

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
    
    # Calculate EMA13 for Elder Ray power calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    bull_power = close - ema_13
    bear_power = ema_13 - close
    
    # Calculate 1d ADX for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = WilderSmoothing(tr, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = WilderSmoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 14+13+13, 20)  # EMA13, ADX (14+13+13 for Wilder smoothing), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX regime filter: trending market (ADX > 25) vs ranging (ADX < 20)
        adx_val = adx_aligned[i]
        trending = adx_val > 25
        ranging = adx_val < 20
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND trending AND volume filter
            if bull_power[i] > 0 and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 (Bear Power positive means bearish) AND trending AND volume filter
            elif bear_power[i] > 0 and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR ADX < 20 (ranging) OR volume normalizes
                if bull_power[i] <= 0 or ranging or not vol_filter:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power <= 0 OR ADX < 20 (ranging) OR volume normalizes
                if bear_power[i] <= 0 or ranging or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_Power_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0