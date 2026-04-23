#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume spike confirmation.
Long when Bear Power < 0 (bulls in control) AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period MA.
Short when Bull Power > 0 (bears in control) AND 1d ADX > 25 (trending) AND 6h volume > 1.5x 20-period MA.
Exit when Elder Power reverses OR ADX < 20 (range) OR volume drops.
Uses 1d HTF for ADX regime to avoid whipsaws in low-volatility environments, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray measures bull/bear power relative to EMA13, ADX filters for trending markets only.
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
    
    # Calculate EMA13 for Elder Ray (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
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
    tr = np.concatenate([[np.nan], tr])  # first TR is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
                else:
                    result[i] = result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_1d = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 14*3)  # EMA13, ADX needs ~42 bars for smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bp = bear_power[i]
        bp_val = bull_power[i]
        adx_val = adx_1d[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Bear Power < 0 (bulls in control) AND ADX > 25 (trending) AND volume filter
            if bp < 0 and adx_val > 25 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power > 0 (bears in control) AND ADX > 25 (trending) AND volume filter
            elif bp_val > 0 and adx_val > 25 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bear Power >= 0 OR ADX < 20 (range) OR volume drops
                if bp >= 0 or adx_val < 20 or not vol_filter:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bull Power <= 0 OR ADX < 20 (range) OR volume drops
                if bp_val <= 0 or adx_val < 20 or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0