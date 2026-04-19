#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w ADX trend filter
# - Long when price breaks above Donchian(20) high + volume > 1.5x 1d average + 1w ADX > 25
# - Short when price breaks below Donchian(20) low + volume > 1.5x 1d average + 1w ADX > 25
# - Exit on opposite Donchian break or ADX < 20 (trend weakening)
# - Position size: 0.25 to balance risk and reward
# - Designed to capture strong trends while avoiding choppy markets
# - Target: 20-50 trades/year to minimize fee drag

name = "4h_Donchian_1dVol_1wADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilde_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1w = wilde_smooth(tr, 14)
    dm_plus_smooth = wilde_smooth(dm_plus, 14)
    dm_minus_smooth = wilde_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = wilde_smooth(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian channels on 4h data
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled to 4h)
        # 1d volume represents ~6 bars of 4h data (24h/4h = 6)
        volume_threshold = 1.5 * vol_ma_1d_aligned[i] / 6.0
        volume_filter = volume[i] > volume_threshold
        
        # ADX filter: trend strength
        strong_trend = adx_1w_aligned[i] > 25
        weak_trend = adx_1w_aligned[i] < 20
        
        if position == 0:
            # Look for long entry: bullish breakout + volume + strong trend
            if close[i] > donchian_high[i] and volume_filter and strong_trend:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish breakout + volume + strong trend
            elif close[i] < donchian_low[i] and volume_filter and strong_trend:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bearish break or weakening trend
            if close[i] < donchian_low[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bullish break or weakening trend
            if close[i] > donchian_high[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals