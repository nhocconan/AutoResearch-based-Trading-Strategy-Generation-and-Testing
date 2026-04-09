#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w ADX regime filter
# In strong trends (ADX > 25): breakout above/below Donchian(20) with volume confirmation
# In weak trends/chop (ADX <= 25): avoid false breakouts, reduce whipsaws
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and minimize fee drag
# Works in bull/bear markets: breakout captures momentum, ADX filter avoids ranging market losses

name = "4h_1d_1w_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Calculate 1d average volume (20-period)
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX(14) for regime filter
    # True Range
    tr1w = np.abs(high_1w[1:] - low_1w[:-1])
    tr2w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3w = np.abs(low_1w[1:] - close_1w[:-1])
    trw = np.concatenate([[np.nan], np.maximum(tr1w, np.maximum(tr2w, tr3w))])
    
    # Directional Movement
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    atr_1w = wilders_smoothing(trw, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_1w = np.where(atr_1w > 0, 100 * plus_dm_smooth / atr_1w, 0.0)
    minus_di_1w = np.where(atr_1w > 0, 100 * minus_dm_smooth / atr_1w, 0.0)
    
    # DX and ADX
    dx_1w = np.where((plus_di_1w + minus_di_1w) > 0, 
                     100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 
                     0.0)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Calculate 4h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align 1d indicators to 4h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    # Align 1w indicators to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 1.5 * avg_volume_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: strong trend when ADX > 25
        strong_trend = adx_1w_aligned[i] > 25.0
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low or trend weakens
            if close[i] < donchian_low[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high or trend weakens
            if close[i] > donchian_high[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian high with volume confirmation and strong trend
            if close[i] > donchian_high[i] and volume_confirmed[i] and strong_trend:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian low with volume confirmation and strong trend
            elif close[i] < donchian_low[i] and volume_confirmed[i] and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals