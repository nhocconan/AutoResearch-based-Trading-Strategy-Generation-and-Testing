#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1h ADX regime filter
# Donchian breakout captures momentum in both bull and bear markets
# 1d volume spike (>2.0x 20-period EMA) confirms institutional participation
# 1h ADX < 25 ensures we only trade in ranging markets to avoid whipsaws
# Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years)
# Works in bull markets (breakout above upper band) and bear markets (breakout below lower band)
# Uses discrete position sizing (0.30) to balance return potential with drawdown control

name = "4h_Donchian20_1dVolumeSpike_1hADX_Range"
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
    
    # 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume EMA20 for spike detection
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # 1h data for ADX regime filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1h timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        def wilders_smoothing(values, period):
            result = np.full_like(values, np.nan, dtype=np.float64)
            if len(values) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                if np.isnan(result[i-1]):
                    result[i] = np.nan
                else:
                    result[i] = (result[i-1] * (period-1) + values[i]) / period
            return result
        
        tr_smoothed = wilders_smoothing(tr, period)
        dm_plus_smoothed = wilders_smoothing(dm_plus, period)
        dm_minus_smoothed = wilders_smoothing(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smoothed / tr_smoothed
        di_minus = 100 * dm_minus_smoothed / tr_smoothed
        
        # DX and ADX
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_1h = calculate_adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Donchian(20) channels on 4h timeframe
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(vol_ema_20_1d_aligned[i]) or np.isnan(adx_1h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: 1d volume > 2.0x 20-period EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20_1d_aligned[i])
        
        # Regime filter: 1h ADX < 25 (ranging market)
        ranging_market = adx_1h_aligned[i] < 25
        
        if position == 0:  # Flat - look for new entries
            if volume_spike and ranging_market:
                if close[i] > donchian_upper[i]:
                    # Long breakout above upper Donchian band
                    signals[i] = 0.30
                    position = 1
                elif close[i] < donchian_lower[i]:
                    # Short breakout below lower Donchian band
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below mid-point of Donchian channel
            mid_point = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price crosses above mid-point of Donchian channel
            mid_point = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals