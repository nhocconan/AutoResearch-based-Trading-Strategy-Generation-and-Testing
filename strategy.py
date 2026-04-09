#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining Donchian(20) breakout with 1d volume confirmation and chop regime filter
# Donchian breakout captures strong momentum moves in both bull and bear markets
# Volume confirmation ensures breakouts have institutional participation
# Chop regime filter (using 1d ADX < 20) avoids false breakouts in ranging markets
# Uses discrete position sizing 0.25 to target 12-37 trades/year and minimize fee drag
# Works in bull markets: follows upward breakouts
# Works in bear markets: follows downward breakouts (shorts)
# Avoids whipsaws in ranging markets via chop filter

name = "12h_1d_donchian_volume_chop_regime_v1"
timeframe = "12h"
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
    
    upper_channel = rolling_max(high_1d, 20)
    lower_channel = rolling_min(low_1d, 20)
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(20, min_periods=20).mean().values
    
    # Calculate 1d ADX(14) for chop regime filter
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = wilders_smoothing(tr, 14)
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 12h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Chop regime filter: ADX < 20 indicates ranging market (avoid breakouts)
        chop_regime = adx_1d_aligned[i] < 20
        trending_regime = adx_1d_aligned[i] >= 20
        
        if position == 1:  # Long position
            # Exit long if price breaks below lower channel OR volume confirmation fails
            if close[i] < lower_channel_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price breaks above upper channel OR volume confirmation fails
            if close[i] > upper_channel_aligned[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only enter in trending regime to avoid false breakouts in chop
            if trending_regime and volume_confirmed:
                # Enter long on upward breakout
                if close[i] > upper_channel_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on downward breakout
                elif close[i] < lower_channel_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals