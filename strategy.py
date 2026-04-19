#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h ADX filter + volume confirmation
# Uses Donchian channels for breakout detection, ADX for trend strength confirmation,
# and volume to filter false breakouts. Works in both bull and bear markets by
# trading breakouts in the direction of the 12h trend (ADX > 25). Target: 15-30 trades/year.
name = "6h_DonchianBreakout_ADX12h_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for multi-timeframe analysis (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h ADX for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = high_12h[0] - close_12h[0]
    tr3[0] = low_12h[0] - close_12h[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def WilderSmooth(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_14 = WilderSmooth(tr_12h, 14)
    dm_plus_14 = WilderSmooth(dm_plus, 14)
    dm_minus_14 = WilderSmooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(tr_14 != 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 != 0, 100 * dm_minus_14 / tr_14, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = WilderSmooth(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h ATR for position sizing
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx_12h_aligned[i]) or \
           np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(atr_6h[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_6h[i]
        
        # Volume filter
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        # ADX filter: trend strength > 25
        trend_filter = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + trend
            if price > donch_high[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + trend
            elif price < donch_low[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below Donchian low or ADX weakens
            if price < donch_low[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above Donchian high or ADX weakens
            if price > donch_high[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals