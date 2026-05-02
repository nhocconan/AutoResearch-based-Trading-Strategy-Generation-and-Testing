#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# 1d ADX > 25 indicates trending market (follow Elder Ray signals)
# 1d ADX < 20 indicates ranging market (fade Elder Ray extremes)
# Volume confirmation (1.5x 20-period average) ensures strong participation
# Discrete position sizing 0.25 to minimize fee churn
# Targets 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime (trend vs range)

name = "6h_ElderRay_1dADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA(13) for Elder Ray
    ema_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_6h  # High - EMA13
    bear_power = ema_6h - low   # EMA13 - Low
    
    # Calculate volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA, ADX and volume MA)
    start_idx = 50  # max(13 for EMA, 34 for ADX, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Trending market (ADX > 25): follow Elder Ray
            if adx > 25:
                # Long: strong bull power + volume spike
                if bull_power[i] > 0 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: strong bear power + volume spike
                elif bear_power[i] > 0 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (ADX < 20): fade Elder Ray extremes
            elif adx < 20:
                # Long: bear power exhaustion (weak bear power) + volume spike
                if bear_power[i] < 0 and volume_spike[i]:  # Bear power negative = bulls in control
                    signals[i] = 0.25
                    position = 1
                # Short: bull power exhaustion (weak bull power) + volume spike
                elif bull_power[i] < 0 and volume_spike[i]:  # Bull power negative = bears in control
                    signals[i] = -0.25
                    position = -1
            # Transition market (20 <= ADX <= 25): no new entries
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bear power expands (market turning bearish) OR ADX strengthens significantly
            if bear_power[i] > 0 or adx > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bull power expands (market turning bullish) OR ADX strengthens significantly
            if bull_power[i] > 0 or adx > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals