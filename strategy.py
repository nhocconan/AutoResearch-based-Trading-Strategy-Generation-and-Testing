#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX chop/trend) and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX regime filter and volume spike (ATR ratio).
- Entry: Long when Bull Power > 0 AND ADX > 25 (trending) AND ATR ratio > 1.5.
         Short when Bear Power < 0 AND ADX > 25 (trending) AND ATR ratio > 1.5.
- Exit: Opposite Elder Ray signal OR ADX < 20 (choppy regime) OR ATR ratio < 1.2 (low volatility).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Elder Ray: Bull Power = Close - EMA13(High), Bear Power = Close - EMA13(Low).
- ADX > 25 filters for trending markets (avoid whipsaws in chop).
- ATR ratio > 1.5 confirms volatility expansion to avoid false signals.
- Works in bull markets (buy strength in uptrend) and bear markets (sell weakness in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Elder Ray signals with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # +DM and -DM
    high_diff = high - np.roll(high, 1)
    low_diff = np.roll(low, 1) - low
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr = atr(high, low, close, period)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx_values

def elder_ray(high, low, close, period):
    """Calculate Elder Ray (Bull Power and Bear Power)."""
    ema_close = ema(close, period)
    bull_power = close - ema_close
    bear_power = close - ema_close  # Same calculation, interpretation differs
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    adx_14 = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 6h Elder Ray (EMA13)
    ema13_close = ema(close, 13)
    bull_power = close - ema13_close
    bear_power = close - ema13_close  # Same calculation, but we interpret negative as bearish
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(adx_14_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Elder Ray signal OR choppy regime OR low volatility
        if position != 0:
            # Exit long: Bear Power >= 0 (weakening bullish) OR ADX < 20 (choppy) OR ATR ratio < 1.2 (low vol)
            if position == 1:
                if bear_power[i] >= 0 or adx_14_aligned[i] < 20 or atr_ratio_aligned[i] < 1.2:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power <= 0 (weakening bearish) OR ADX < 20 (choppy) OR ATR ratio < 1.2 (low vol)
            elif position == -1:
                if bull_power[i] <= 0 or adx_14_aligned[i] < 20 or atr_ratio_aligned[i] < 1.2:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray with trend filter and volume confirmation
        if position == 0:
            # Long: Bull Power > 0 (bullish) AND ADX > 25 (trending) AND ATR ratio > 1.5 (vol spike)
            if bull_power[i] > 0 and adx_14_aligned[i] > 25 and atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish) AND ADX > 25 (trending) AND ATR ratio > 1.5 (vol spike)
            elif bear_power[i] < 0 and adx_14_aligned[i] > 25 and atr_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_RegimeFilter_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0