#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w ADX regime filter
# - Uses 6h Elder Ray (Bull Power = Close - EMA13, Bear Power = EMA13 - Close) to measure bull/bear strength
# - 1w ADX(14) > 25 indicates strong trend (regime filter) - avoids whipsaws in ranging markets
# - Long when Bull Power > 0 and 1w ADX > 25; Short when Bear Power > 0 and 1w ADX > 25
# - Position size 0.25 to manage drawdown in volatile markets
# - Works in bull trends via strong Bull Power, in bear trends via strong Bear Power
# - Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Elder Ray captures momentum shifts, ADX regime filter ensures trades only in trending conditions

name = "6h_1w_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for regime filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Rest is Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1w = WilderSmoothing(tr_1w, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus_1w = np.where(atr_1w > 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus_1w = np.where(atr_1w > 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx_1w = np.where((di_plus_1w + di_minus_1w) > 0,
                     np.abs(di_plus_1w - di_minus_1w) / (di_plus_1w + di_minus_1w) * 100, 0)
    adx_1w = WilderSmoothing(dx_1w, 14)
    
    # Align 1w ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema_13  # Close - EMA13
    bear_power = ema_13 - close  # EMA13 - Close
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1w_aligned[i]) or adx_1w_aligned[i] < 25 or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Bull Power turns negative or ADX weakens
            if bull_power[i] <= 0 or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Bear Power turns negative or ADX weakens
            if bear_power[i] <= 0 or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries: strong Bull/Bear Power with trending regime
            if bull_power[i] > 0 and adx_1w_aligned[i] > 25:
                position = 1
                signals[i] = 0.25
            elif bear_power[i] > 0 and adx_1w_aligned[i] > 25:
                position = -1
                signals[i] = -0.25
    
    return signals