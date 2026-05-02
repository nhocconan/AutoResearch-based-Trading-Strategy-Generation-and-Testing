#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1w ADX regime filter
# Uses 4h primary timeframe for Donchian channel breakout signals (long on upper band, short on lower)
# 1d volume confirmation (2.0x 20-period average) ensures strong participation
# 1w ADX(14) > 25 filters for trending markets only, avoiding choppy conditions
# Discrete position sizing (0.30) balances profit potential with fee drag minimization
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# Donchian provides clear structure, volume confirms conviction, ADX ensures trend strength
# Works in both bull and bear markets by only trading when ADX confirms trending regime

name = "4h_Donchian20_1dVolumeSpike_1wADX_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike (2.0x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    high_1w = pd.Series(df_1w['high'])
    low_1w = pd.Series(df_1w['low'])
    close_1w = pd.Series(df_1w['close'])
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w.shift(1))
    tr3 = np.abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = np.where((high_1w - high_1w.shift(1)) > (low_1w.shift(1) - low_1w), 
                       np.maximum(high_1w - high_1w.shift(1), 0), 0)
    dm_minus = np.where((low_1w.shift(1) - low_1w) > (high_1w - high_1w.shift(1)), 
                        np.maximum(low_1w.shift(1) - low_1w, 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smooth(series, period):
        result = np.full_like(series, np.nan, dtype=float)
        if len(series) < period:
            return result
        result[period-1] = np.nansum(series[:period])
        for i in range(period, len(series)):
            result[i] = result[i-1] - (result[i-1] / period) + series[i]
        return result
    
    tr_14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_14 != 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 != 0, (dm_minus_14 / tr_14) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilder_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and HTF calculations)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band + volume spike + ADX > 25
            long_entry = (close[i] > donchian_upper[i] and 
                         volume_spike_1d_aligned[i] and 
                         adx_aligned[i] > 25)
            
            # Short: price breaks below Donchian lower band + volume spike + ADX > 25
            short_entry = (close[i] < donchian_lower[i] and 
                          volume_spike_1d_aligned[i] and 
                          adx_aligned[i] > 25)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or ADX weakens (< 20)
            if close[i] < donchian_lower[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or ADX weakens (< 20)
            if close[i] > donchian_upper[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals