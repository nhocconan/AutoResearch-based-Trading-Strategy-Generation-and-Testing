#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
# - Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
# - Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and volume spike
# - Short when Bear Power > 0, Bull Power < 0, ADX > 25, and volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in both bull and bear markets by using ADX regime filter to only trade strong trends

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Bull Power and Bear Power
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = ema_13_1d - low_1d
    
    # 1d ADX(14) for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_14_1d = WilderSmoothing(tr, 14)
    dm_plus_14_1d = WilderSmoothing(dm_plus, 14)
    dm_minus_14_1d = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus_14_1d = 100 * dm_plus_14_1d / atr_14_1d
    di_minus_14_1d = 100 * dm_minus_14_1d / atr_14_1d
    
    # DX and ADX
    dx_14_1d = 100 * np.abs(di_plus_14_1d - di_minus_14_1d) / (di_plus_14_1d + di_minus_14_1d)
    adx_14_1d = WilderSmoothing(dx_14_1d, 14)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    
    # Align all HTF indicators to LTF
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: when trend weakens or power reverses
            if (adx_14_1d_aligned[i] < 20 or  # Trend weakening
                bull_power_1d_aligned[i] <= 0 or  # Bull Power gone
                bear_power_1d_aligned[i] >= 0):   # Bear Power appears
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: when trend weakens or power reverses
            if (adx_14_1d_aligned[i] < 20 or  # Trend weakening
                bear_power_1d_aligned[i] <= 0 or  # Bear Power gone
                bull_power_1d_aligned[i] >= 0):   # Bull Power appears
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with trend and volume filters
            if vol_spike_1d_aligned[i] and adx_14_1d_aligned[i] > 25:
                # Long signal: Bull Power > 0, Bear Power < 0, strong trend
                if (bull_power_1d_aligned[i] > 0 and 
                    bear_power_1d_aligned[i] < 0):
                    position = 1
                    signals[i] = 0.25
                # Short signal: Bear Power > 0, Bull Power < 0, strong trend
                elif (bear_power_1d_aligned[i] > 0 and 
                      bull_power_1d_aligned[i] < 0):
                    position = -1
                    signals[i] = -0.25
    
    return signals