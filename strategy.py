#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout + 1d ADX trend filter + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for trend strength (trending if ADX > 25, ranging if ADX < 20).
- Bollinger Bands(20,2) on 6h: long on break above upper band when BB width at 20-period low (squeeze breakout).
- Short on break below lower band under same squeeze condition.
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter weak breakouts.
- Signal size: 0.25 discrete to balance return and drawdown control.
- Designed to work in both bull and bear markets by using ADX regime filter to avoid false breakouts in ranging markets.
- Uses discrete position sizing to minimize fee churn and respects 6h timeframe trade frequency limits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Bollinger Bands calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need for BB and volume MA
        return np.zeros(n)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need for ADX calculation
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20,2)
    close_6h = df_6h['close'].values
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # BB width 20-period low for squeeze detection (lowest width in last 20 periods)
    bb_width_low = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    bb_squeeze = bb_width <= bb_width_low  # True when at 20-period low width
    
    # Align 6h indicators to primary 6h timeframe (no additional delay needed for BB)
    upper_bb_aligned = align_htf_to_ltf(prices, df_6h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_6h, lower_bb)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_6h, bb_squeeze.astype(float))
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                       np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe with 1-bar delay (wait for 1d close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA(20) for confirmation
    vol_ma_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30, 20, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(bb_squeeze_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and squeeze breakout
            vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
            squeeze_active = bb_squeeze_aligned[i] > 0.5  # True when in squeeze
            
            # Determine 1d trend: strong trend if ADX > 25
            strong_trend = adx_aligned[i] > 25
            
            # Long: price breaks above upper BB AND in squeeze AND strong trend AND volume confirmed
            if curr_high > upper_bb_aligned[i] and squeeze_active and strong_trend and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower BB AND in squeeze AND strong trend AND volume confirmed
            elif curr_low < lower_bb_aligned[i] and squeeze_active and strong_trend and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on price re-entering Bollinger Bands (mean reversion)
            if curr_low < upper_bb_aligned[i] and curr_high > lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on price re-entering Bollinger Bands (mean reversion)
            if curr_low < upper_bb_aligned[i] and curr_high > lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerSqueeze_Breakout_1dADX_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0