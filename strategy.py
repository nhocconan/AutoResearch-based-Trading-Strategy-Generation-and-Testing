#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian channel breakouts capture trends, while volume and ADX filters
# ensure momentum and reduce false signals. Works in both bull and bear markets by
# only taking breakouts in the direction of the trend (ADX > 25). Volume confirms
# institutional participation. Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d 50 EMA for trend filter
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    ema_50_1d = close_series_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) for trend strength
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM and -DM
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low or trend weakens
            if close[i] <= donchian_low[i] or adx[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or trend weakens
            if close[i] >= donchian_high[i] or adx[i] < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and strong trend
            if (high[i] > donchian_high[i] or close[i] > donchian_high[i]) and \
               volume[i] > vol_ma[i] * 1.5 and \
               adx[i] > 25 and \
               close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and strong trend
            elif (low[i] < donchian_low[i] or close[i] < donchian_low[i]) and \
                 volume[i] > vol_ma[i] * 1.5 and \
                 adx[i] > 25 and \
                 close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals