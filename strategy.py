#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian_Breakout_Volume
Hypothesis: Combines ADX regime filter with Donchian channel breakout and volume confirmation on 6h timeframe.
Enter long when price breaks above Donchian(20) upper band AND ADX > 25 (trending) AND volume > 1.5 * 20-period average.
Enter short when price breaks below Donchian(20) lower band AND ADX > 25 (trending) AND volume > 1.5 * 20-period average.
Exit when price returns to Donchian midpoint OR ADX < 20 (range regime) OR opposite breakout occurs.
Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades in strong trends.
Designed to work in both bull and bear markets by only taking trades in strong trending regimes (ADX>25).
Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX(14) on 6h data
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nanmean(data[:period])  # Seed with simple average
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di_smoothed = wilders_smoothing(plus_dm, period)
    minus_di_smoothed = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    dx = np.where((plus_di_smoothed + minus_di_smoothed) != 0,
                  100 * np.abs(plus_di_smoothed - minus_di_smoothed) / (plus_di_smoothed + minus_di_smoothed),
                  0)
    adx = wilders_smoothing(dx, period)
    
    # Donchian Channel(20) on 6h data
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle_band = (upper_band + lower_band) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need ADX (14*3=42), Donchian (20), volume avg (20), 1d EMA (50)
    start_idx = max(42, 20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        adx_val = adx[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        ema_1d_val = ema_50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with ADX > 25 (trending) AND volume
            # Long: price breaks above upper band AND ADX > 25 AND volume
            long_condition = (close_val > upper) and (adx_val > 25) and vol_conf
            # Short: price breaks below lower band AND ADX > 25 AND volume
            short_condition = (close_val < lower) and (adx_val > 25) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to middle OR ADX < 20 (range) OR opposite breakout
            exit_condition = (close_val <= middle) or (adx_val < 20) or (close_val < lower)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to middle OR ADX < 20 (range) OR opposite breakout
            exit_condition = (close_val >= middle) or (adx_val < 20) or (close_val > upper)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_Regime_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0