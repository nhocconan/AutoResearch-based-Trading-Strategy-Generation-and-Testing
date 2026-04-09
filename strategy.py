#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets, while ADX > 25 filters for strong trends
# Volume confirmation ensures breakouts have participation. Works in bull/bear markets as breakouts
# occur in both directions. Uses discrete sizing 0.25 to target ~50-150 trades over 4 years.
# Uses proper MTF data loading with get_htf_data called once before loop.

name = "6h_12h_1d_donchian_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.zeros_like(close_12h)
    
    # Load 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high_12h = rolling_max(high_12h, 20)
    donchian_low_12h = rolling_min(low_12h, 20)
    
    # Calculate 1d ADX (14-period)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range components
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_1d = wilders_smoothing(plus_dm, 14)
    minus_dm_1d = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di_1d = 100 * plus_dm_1d / atr_1d
    minus_di_1d = 100 * minus_dm_1d / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, dx_1d, 0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 1d average volume (20-period)
    vol_s_1d = pd.Series(volume_1d := df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros_like(close_1d))
    avg_vol_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 6h timeframe
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled to 6h)
        # Approximate 6h average volume as 1d average volume / 4 (since 4x 6h in 1d)
        vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = volume[i] > 1.5 * (avg_vol_1d_aligned[i] / 4) if not np.isnan(vol_ma_20_6h[i]) else False
        
        # ADX trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit long if price falls below Donchian low or trend weakens
            if close[i] < donchian_low_12h_aligned[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above Donchian high or trend weakens
            if close[i] > donchian_high_12h_aligned[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on Donchian high breakout with volume and trend confirmation
            if close[i] > donchian_high_12h_aligned[i] and volume_confirmed and strong_trend:
                position = 1
                signals[i] = 0.25
            # Enter short on Donchian low breakdown with volume and trend confirmation
            elif close[i] < donchian_low_12h_aligned[i] and volume_confirmed and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals