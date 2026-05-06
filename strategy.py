#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Keltner channels with ADX trend filter and volume confirmation
# Keltner channels (EMA-based) adapt to volatility better than fixed % bands
# ADX > 25 indicates trending market for breakout trades; ADX < 20 indicates ranging for mean reversion
# Volume > 1.5x average confirms institutional participation
# Works in bull/bear: trend following in strong moves, mean reversion in low volatility regimes
# Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_Keltner_ADX_VolumeFilter_v1"
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
    
    # Calculate daily EMA for Keltner center line
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(20) for Keltner center
    ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily ATR(10) for Keltner width
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels: EMA(20) ± 2 * ATR(10)
    upper = ema_20 + (2 * atr_10)
    lower = ema_20 - (2 * atr_10)
    
    # Align daily Keltner levels to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    center_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # ADX(14) from daily data for trend strength
    # Calculate +DM, -DM, TR
    high_diff = pd.Series(df_1d['high']).diff()
    low_diff = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr_atr = pd.concat([
        pd.Series(df_1d['high']).diff().abs(),
        (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs(),
        (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    ], axis=1).max(axis=1)
    
    # Smoothed values
    atr_14 = tr_atr.ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * (plus_dm_smooth / atr_14)
    minus_di = 100 * (minus_dm_smooth / atr_14)
    
    # DX and ADX
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(center_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: ADX > 25 (trending) + price breaks above upper Keltner + volume
            if adx_aligned[i] > 25 and close[i] > upper_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: ADX > 25 (trending) + price breaks below lower Keltner + volume
            elif adx_aligned[i] > 25 and close[i] < lower_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Long mean reversion: ADX < 20 (ranging) + price touches lower Keltner + volume
            elif adx_aligned[i] < 20 and close[i] < lower_aligned[i] * 1.002 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short mean reversion: ADX < 20 (ranging) + price touches upper Keltner + volume
            elif adx_aligned[i] < 20 and close[i] > upper_aligned[i] * 0.998 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: ADX drops below 20 (trend weakening) or price returns to center
            if adx_aligned[i] < 20 or close[i] > center_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: ADX drops below 20 (trend weakening) or price returns to center
            if adx_aligned[i] < 20 or close[i] < center_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals