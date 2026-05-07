#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d volume surge and ADX trend filter.
# Long when price breaks above upper Bollinger Band (20,2) AND 1d volume > 1.5x 20-day average AND 1d ADX > 25.
# Short when price breaks below lower Bollinger Band (20,2) AND 1d volume > 1.5x 20-day average AND 1d ADX > 25.
# Exit when price returns to the 20-period SMA (middle Bollinger Band).
# This strategy captures low volatility breakouts with institutional volume and trend confirmation.
# Bollinger squeeze identifies compression before expansion, volume surge confirms participation,
# and ADX ensures we only trade in trending conditions to avoid whipsaws in ranging markets.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by requiring ADX > 25 (strong trend) and volume surge.

name = "4h_BollingerSqueeze_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Bollinger Bands (20,2) on 4h close
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # 1d volume surge: current volume > 1.5x 20-day average
    vol_ma20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_surge = df_1d['volume'].values > (1.5 * vol_ma20_1d)
    volume_surge_aligned = align_htf_to_ltf(prices, df_1d, volume_surge)
    
    # 1d ADX (14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Sufficient warmup for Bollinger Bands and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(sma20[i]) or np.isnan(volume_surge_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB, volume surge, ADX > 25
            long_cond = (close[i] > upper_bb[i]) and volume_surge_aligned[i] and (adx_aligned[i] > 25)
            # Short conditions: price breaks below lower BB, volume surge, ADX > 25
            short_cond = (close[i] < lower_bb[i]) and volume_surge_aligned[i] and (adx_aligned[i] > 25)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle Bollinger Band (SMA20)
            if close[i] >= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle Bollinger Band (SMA20)
            if close[i] <= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals