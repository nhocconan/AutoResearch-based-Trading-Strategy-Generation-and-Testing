#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and 1w ADX regime filter
# Donchian(20) breakout captures strong moves; volume confirmation ensures institutional participation
# 1w ADX > 25 = trending regime (follow breakout direction); ADX < 20 = ranging regime (fade breakouts)
# Works in bull/bear markets: breakout follows trend in strong trends, fades false breakouts in ranging markets
# Discrete position sizing 0.25 targets ~20-50 trades/year to minimize fee drag

name = "4h_1d_1w_donchian_volume_adx_regime_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window, min_periods=window).min().values
    
    donchian_high_20 = rolling_max(high_1d, 20)
    donchian_low_20 = rolling_min(low_1d, 20)
    
    # Calculate 1d volume average (20-period)
    volume_ma_20 = pd.Series(volume_1d).rolling(20, min_periods=20).mean().values
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14) using Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = wilders_smoothing(tr, period)
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align 1d indicators to 4h timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # Align 1w ADX to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(volume_ma_20_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_ma_20_aligned[i]
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if price breaks below Donchian low
                if close[i] <= donchian_low_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if price returns to midpoint
                midpoint = (donchian_high_20_aligned[i] + donchian_low_20_aligned[i]) / 2
                if close[i] >= midpoint:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if price breaks above Donchian high
                if close[i] >= donchian_high_20_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if price returns to midpoint
                midpoint = (donchian_high_20_aligned[i] + donchian_low_20_aligned[i]) / 2
                if close[i] <= midpoint:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime and volume_confirm:
                # Follow breakout direction in trending market with volume confirmation
                if close[i] >= donchian_high_20_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] <= donchian_low_20_aligned[i]:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime and volume_confirm:
                # Fade breakouts in ranging market with volume confirmation (fade false breakouts)
                if close[i] <= donchian_low_20_aligned[i]:
                    position = 1  # Long false breakdown
                    signals[i] = 0.25
                elif close[i] >= donchian_high_20_aligned[i]:
                    position = -1  # Short false breakout
                    signals[i] = -0.25
    
    return signals