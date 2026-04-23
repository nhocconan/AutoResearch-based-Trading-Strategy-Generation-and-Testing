#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) combined with 1d ADX regime filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
- Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA
- Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND 1d ADX > 25 AND volume > 1.5x 20-period MA
- Exit when Elder Power weakens (Bull Power <= 0 for longs, Bear Power <= 0 for shorts) OR ADX < 20 (range) OR volume drops
- Uses 1d HTF for ADX trend regime to avoid whipsaws in low ADX environments, volume confirmation for momentum.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull (strong ADX + Bull Power) and bear (strong ADX + Bear Power) regimes.
- Avoids ranging markets via ADX < 20 filter.
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
    
    # Calculate EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Calculate 1d ADX for trend regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_period = 14
    atr_1d = WilderSmoothing(tr, atr_period)
    dm_plus_smooth = WilderSmoothing(dm_plus, atr_period)
    dm_minus_smooth = WilderSmoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = WilderSmoothing(dx, atr_period)  # ADX is smoothed DX
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30 + atr_period*2, 20)  # EMA13, ADX calculation, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # ADX regime filter: trending market (ADX > 25) vs ranging (ADX < 20)
        adx_val = adx_1d_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND trending AND volume filter
            if bull_power[i] > 0 and bear_power[i] < 0 and is_trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND trending AND volume filter
            elif bear_power[i] > 0 and bull_power[i] < 0 and is_trending and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 (momentum weakening) OR ADX < 20 (ranging) OR volume drops
                if bull_power[i] <= 0 or is_ranging or not vol_filter:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power <= 0 (momentum weakening) OR ADX < 20 (ranging) OR volume drops
                if bear_power[i] <= 0 or is_ranging or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_ADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0