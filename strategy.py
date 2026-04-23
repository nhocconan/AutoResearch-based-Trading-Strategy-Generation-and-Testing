#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX Regime Filter
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (6h timeframe)
- 1d ADX(14) defines market regime: ADX > 25 = trending (trade with momentum), ADX < 20 = ranging (fade extremes)
- In trending regime (ADX > 25): Go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
- In ranging regime (ADX < 20): Go long when Bear Power < -0.5*ATR(6) and turning up, short when Bull Power > 0.5*ATR(6) and turning down
- Volume confirmation: require volume > 1.5x 20-period average to avoid low-activity false signals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by adapting to regime: momentum in trends, mean reversion in ranges
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
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 6h ATR(14) for volatility normalization in ranging regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = tr1_1d[0]
    tr3_1d[0] = tr1_1d[0]
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            result[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1d = wilder_smoothing(tr_1d, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilder_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 14)  # for ADX, volume MA, and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry logic
            if adx_1d_aligned[i] > 25:  # Trending regime - trade with momentum
                # Long: Bull Power positive and rising (bullish momentum)
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power negative and falling (bearish momentum)
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
            elif adx_1d_aligned[i] < 20:  # Ranging regime - fade extremes
                # Long: Bear Power significantly negative and turning up
                if bear_power[i] < -0.5 * atr_14[i] and bear_power[i] > bear_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short: Bull Power significantly positive and turning down
                elif bull_power[i] > 0.5 * atr_14[i] and bull_power[i] < bull_power[i-1]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit logic: regime change or signal exhaustion
            exit_signal = False
            
            if position == 1:  # Long position
                if adx_1d_aligned[i] > 25:  # Still trending - exit when momentum fades
                    if bull_power[i] <= 0 or bull_power[i] < bull_power[i-1]:
                        exit_signal = True
                else:  # Regime changed to ranging - exit when overextended
                    if bull_power[i] < 0.2 * atr_14[i]:
                        exit_signal = True
            elif position == -1:  # Short position
                if adx_1d_aligned[i] > 25:  # Still trending - exit when momentum fades
                    if bear_power[i] >= 0 or bear_power[i] > bear_power[i-1]:
                        exit_signal = True
                else:  # Regime changed to ranging - exit when overextended
                    if bear_power[i] > -0.2 * atr_14[i]:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0