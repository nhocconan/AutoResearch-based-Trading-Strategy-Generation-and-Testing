#!/usr/bin/env python3
# 6h_elder_ray_regime_v1
# Hypothesis: 6h strategy using Elder Ray (Bull/Bear Power) with ADX regime filter.
# Long when Bull Power > 0, Bear Power < 0, and ADX > 25 (strong trend).
# Short when Bear Power < 0, Bull Power > 0, and ADX > 25.
# Uses 1d EMA(13) for power calculation and 1w ADX(14) for regime.
# Discrete position sizing (0.25) to limit fees. Target: 12-37 trades/year (50-150 total over 4 years).
# Works in bull/bear: ADX ensures we only trade strong trends, Elder Ray captures momentum with trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA(13) for Elder Ray calculation
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema13
    
    # Multi-timeframe: 1w ADX(14) for regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_1w = wilders_smoothing(tr_1w, 14)
    dm_plus_1w = wilders_smoothing(dm_plus, 14)
    dm_minus_1w = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus_1w = np.where(atr_1w != 0, (dm_plus_1w / atr_1w) * 100, 0)
    di_minus_1w = np.where(atr_1w != 0, (dm_minus_1w / atr_1w) * 100, 0)
    
    # DX and ADX
    dx_1w = np.where((di_plus_1w + di_minus_1w) != 0,
                     np.abs(di_plus_1w - di_minus_1w) / (di_plus_1w + di_minus_1w) * 100, 0)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Align HTF indicators to LTF
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 or Bear Power >= 0 (momentum weakening) or trend weakens
            if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 or Bull Power <= 0 (momentum weakening) or trend weakens
            if bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Elder Ray signals with trend confirmation
            bullish = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and strong_trend
            bearish = bear_power_aligned[i] < 0 and bull_power_aligned[i] > 0 and strong_trend
            
            if bullish:
                position = 1
                signals[i] = 0.25
            elif bearish:
                position = -1
                signals[i] = -0.25
    
    return signals