#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + ADX trend filter with volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (measures bull/bear strength)
- ADX > 25 indicates strong trend (filters choppy markets)
- Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume > 1.5x 20-period avg
- Short: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 AND volume > 1.5x 20-period avg
- Exit: Elder Ray divergence (Bull Power <= 0 for long, Bear Power <= 0 for short) OR ADX < 20 (trend weakens)
- Uses 12h HTF for EMA13 calculation to reduce noise
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy strength in uptrend) and bear (sell strength in downtrend)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA13 for Elder Ray (less noisy than 6h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    ema_13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_13_12h)
    
    # Calculate Elder Ray components
    bull_power = high_12h - ema_13_12h  # Measures bull strength
    bear_power = ema_13_12h - low_12h   # Measures bear strength
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # Calculate ADX from 12h data for trend strength
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_12h = np.concatenate([[close_12h[0]], close_12h[:-1]])
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - prev_close_12h)
    tr3 = np.abs(low_12h - prev_close_12h)
    tr_12h = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # +DM = max(high - prev_high, 0) if > max(prev_low - low, 0) else 0
    prev_high_12h = np.concatenate([[high_12h[0]], high_12h[:-1]])
    prev_low_12h = np.concatenate([[low_12h[0]], low_12h[:-1]])
    plus_dm = high_12h - prev_high_12h
    minus_dm = prev_low_12h - low_12h
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: prev_result * (1 - 1/period) + current * (1/period)
            alpha = 1.0 / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr_12h = wilder_smooth(tr_12h, 14)
    plus_di_12h = 100 * wilder_smooth(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilder_smooth(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilder_smooth(dx_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 14+14+14)  # EMA13, vol MA, ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_13_12h_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 AND volume confirmation
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                adx_12h_aligned[i] > 25 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 AND volume confirmation
            elif (bear_power_aligned[i] > 0 and 
                  bull_power_aligned[i] < 0 and 
                  adx_12h_aligned[i] > 25 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (weakening bulls) OR ADX < 20 (trend weakening)
            if bull_power_aligned[i] <= 0 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (weakening bears) OR ADX < 20 (trend weakening)
            if bear_power_aligned[i] <= 0 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0