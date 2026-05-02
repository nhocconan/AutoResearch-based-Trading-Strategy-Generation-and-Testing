#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# Uses Donchian channel for structure, 1d volume for confirmation, 1w ADX>25 for trend strength
# Works in both bull and bear markets by following higher timeframe trend
# Target: 100-180 total trades over 4 years (25-45/year) to balance edge and fees
# Discrete position sizing: 0.30 (30% of capital) for controlled risk

name = "4h_Donchian20_1dVolumeSpike_1wADX25_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Calculate 1d volume spike (2.0x 20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 1w ADX for trend filter (ADX > 25 = strong trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    atr_1w = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, (dm_plus_smooth / atr_1w) * 100, 0)
    di_minus = np.where(atr_1w != 0, (dm_minus_smooth / atr_1w) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1w = wilders_smoothing(dx, atr_period)
    adx_strong = adx_1w > 25
    adx_strong_aligned = align_htf_to_ltf(prices, df_1w, adx_strong.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 20, 14+14)  # Donchian, volume, ADX
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(adx_strong_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper channel with volume spike AND strong 1w uptrend (DI+ > DI-)
            if (close[i] > upper_channel[i] and 
                vol_spike_1d_aligned[i] > 0.5 and 
                adx_strong_aligned[i] > 0.5):
                # Additional trend filter: DI+ > DI- for long
                # We need DI values aligned - recalculate or use close position vs EMA as proxy
                # Simplified: use price > 20 EMA for trend direction
                ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                if not np.isnan(ema_20[i]) and close[i] > ema_20[i]:
                    signals[i] = 0.30
                    position = 1
            # Short entry: price breaks below Donchian lower channel with volume spike AND strong 1w downtrend (DI- > DI+)
            elif (close[i] < lower_channel[i] and 
                  vol_spike_1d_aligned[i] > 0.5 and 
                  adx_strong_aligned[i] > 0.5):
                # Additional trend filter: DI- > DI+ for short
                ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                if not np.isnan(ema_20[i]) and close[i] < ema_20[i]:
                    signals[i] = -0.30
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian lower channel OR ADX weakens (<20)
            if close[i] < lower_channel[i] or adx_strong_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper channel OR ADX weakens (<20)
            if close[i] > upper_channel[i] or adx_strong_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals