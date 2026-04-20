#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# - Long when price breaks above 20-period Donchian high + ADX(14) > 25 + volume > 1.5x average
# - Short when price breaks below 20-period Donchian low + ADX(14) > 25 + volume > 1.5x average
# - Exit when price crosses the Donchian midline (average of upper/lower bands)
# - Uses weekly ADX for trend strength to avoid whipsaws in ranging markets
# - Volume confirmation ensures breakouts have conviction
# - Designed for 12h timeframe with selective entries to stay within trade limits
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on weekly timeframe
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period]) if period > 1 else data[0]
        # Subsequent values: smoothed = prev * (period-1)/period + current/period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (period-1)/period + data[i]/period
        return result
    
    atr_period = 14
    atr_1w = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = wilders_smoothing(dx, atr_period)
    
    # Align weekly ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian channels on 12h data
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    donchian_period = 20
    upper_band = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_band = (upper_band + lower_band) / 2
    
    # Volume average for confirmation
    vol_avg = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(adx_12h[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume_12h[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long entry: price breaks above upper band + ADX > 25 + volume confirmation
            if close_12h[i] > upper_band[i] and adx_12h[i] > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + ADX > 25 + volume confirmation
            elif close_12h[i] < lower_band[i] and adx_12h[i] > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle band
            if close_12h[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle band
            if close_12h[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wADXFilter_Volume"
timeframe = "12h"
leverage = 1.0