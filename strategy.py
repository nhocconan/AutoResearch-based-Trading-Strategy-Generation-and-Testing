#!/usr/bin/env python3
"""
Hypothesis: 1-day Bollinger Band squeeze breakout with 1-week ADX trend filter and volume confirmation.
Long when price breaks above upper BB during low volatility (BBW < 20th percentile) with rising weekly ADX > 25 and volume spike.
Short when price breaks below lower BB during low volatility with rising weekly ADX > 25 and volume spike.
Exit when price returns to middle Bollinger Band (20-day SMA).
Designed to capture explosive moves after consolidation periods, works in both bull and bear markets by filtering with weekly trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Bollinger Bands - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    bb_width = (upper_bb - lower_bb) / sma20 * 100  # Percentage
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct20 = bb_width_series.expanding(min_periods=20).quantile(0.20).values
    
    # Load 1-week data for ADX trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 1-day timeframe
    sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_pct20_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(sma20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(bb_width_pct20_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width below 20th percentile
        squeeze = bb_width[i] < bb_width_pct20_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Weekly ADX rising (current > previous)
        adx_rising = adx[i] > adx[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Price breaks above upper BB during squeeze with rising ADX > 25 and volume spike
            if (squeeze and close[i] > upper_bb_aligned[i] and 
                adx_aligned[i] > 25 and adx_rising and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB during squeeze with rising ADX > 25 and volume spike
            elif (squeeze and close[i] < lower_bb_aligned[i] and 
                  adx_aligned[i] > 25 and adx_rising and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle Bollinger Band (20-day SMA)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below SMA20
                if close[i] < sma20_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above SMA20
                if close[i] > sma20_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Bollinger_Squeeze_ADXTrend_Volume"
timeframe = "1d"
leverage = 1.0