#!/usr/bin/env python3
"""
6h_ADX_WilliamsAlligator_Regime
Hypothesis: Combines ADX trend strength with Williams Alligator (SMAs) on 6h to identify trending regimes, then uses 1d EMA cross for entry timing. Works in bull/bear by only taking trades in direction of higher timeframe trend (1d EMA50 > EMA200 = long only, < = short only). Discrete sizing 0.25 targets 12-37 trades/year. Uses ATR-based stoploss for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams Alligator on 6h: Jaw(13), Teeth(8), Lips(5) SMAs
    close_series = pd.Series(close)
    jaw = close_series.rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = close_series.rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = close_series.rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Calculate ADX(14) on 6h
    # True Range
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - high_shift
    down_move = low_shift - low
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period median
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 200 for EMA200, 14 for ADX/ATR, 13 for Alligator jaw, 20 for volume
    start_idx = max(200, 14, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(adx[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        ema_200_val = ema_200_1d_aligned[i]
        adx_val = adx[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Determine trend direction from 1d EMA cross
        uptrend_1d = ema_50_val > ema_200_val
        downtrend_1d = ema_50_val < ema_200_val
        
        # Williams Alligator signals: Jaw > Teeth > Lips = uptrend, reverse = downtrend
        alligator_long = jaw_val > teeth_val and teeth_val > lips_val
        alligator_short = jaw_val < teeth_val and teeth_val < lips_val
        
        if position == 0:
            # Flat - look for entry
            # Long: ADX > 25 (strong trend) + Alligator aligned up + 1d uptrend + volume spike
            long_entry = (adx_val > 25) and alligator_long and uptrend_1d and vol_spike
            # Short: ADX > 25 + Alligator aligned down + 1d downtrend + volume spike
            short_entry = (adx_val > 25) and alligator_short and downtrend_1d and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend weakening, Alligator reversal, or ATR stop
            stop_price = entry_price - 2.5 * atr_val
            # Exit if trend weakens (ADX < 20) or Alligator reverses or price hits stop
            if (adx_val < 20) or not alligator_long or close_val < stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend weakening, Alligator reversal, or ATR stop
            stop_price = entry_price + 2.5 * atr_val
            # Exit if trend weakens (ADX < 20) or Alligator reverses or price hits stop
            if (adx_val < 20) or not alligator_short or close_val > stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_WilliamsAlligator_Regime"
timeframe = "6h"
leverage = 1.0