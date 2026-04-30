#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, 1d ADX > 25 (trending), and volume > 1.5x 20-bar avg.
# Short when Bear Power < 0, Bull Power > 0, 1d ADX > 25, and volume > 1.5x 20-bar avg.
# Exit when Elder Power signals reverse or ADX < 20 (range regime).
# Uses 1d ADX to filter for trending markets only, avoiding whipsaws in chop.
# Elder Ray captures underlying bull/bear strength via EMA(13) relationship.
# Volume confirmation reduces false signals.
# Discrete position sizing at ±0.25 to balance performance and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_ElderRay_1dADX_TrendFilter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ with Wilder's smoothing (alpha = 1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Elder Ray (Bull Power, Bear Power) on 6h
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA13 and 1d ADX
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_adx = adx_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = curr_adx > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0, Bear Power < 0, trending, volume spike
            if (curr_bull_power > 0 and 
                curr_bear_power < 0 and 
                is_trending and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, trending, volume spike
            elif (curr_bear_power < 0 and 
                  curr_bull_power > 0 and 
                  is_trending and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions:
            # 1. Elder Power signals reverse (Bull Power <= 0 or Bear Power >= 0)
            # 2. ADX drops below 20 (range regime)
            if (curr_bull_power <= 0 or 
                curr_bear_power >= 0 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Elder Power signals reverse (Bear Power >= 0 or Bull Power <= 0)
            # 2. ADX drops below 20 (range regime)
            if (curr_bear_power >= 0 or 
                curr_bull_power <= 0 or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals