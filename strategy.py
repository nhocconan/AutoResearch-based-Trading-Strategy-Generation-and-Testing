#!/usr/bin/env python3
"""
Hypothesis: 4h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above upper BB AND ADX > 25 (trending) AND volume > 1.5x average.
Short when price breaks below lower BB AND ADX > 25 (trending) AND volume > 1.5x average.
Exit when price returns to middle BB (20-period SMA) or volume drops below average.
Bollinger Band squeeze identifies low volatility periods primed for breakout.
1d ADX > 25 ensures trading only in strong trending regimes (works in bull/bear markets).
Volume confirmation avoids false breakouts.
Designed for 4h timeframe targeting 75-200 total trades over 4 years to minimize fee drag.
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
    
    # Load 1d data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([np.nan]), tr])  # align with 1d indices
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([np.array([0.0]), dm_plus])
    dm_minus = np.concatenate([np.array([0.0]), dm_minus])
    
    # Smooth TR, DM+ (14-period Wilder's smoothing = EMA with alpha=1/14)
    def wilders_smooth(data, period):
        alpha = 1.0 / period
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.nanmean(data[:period])  # seed with simple average
        for i in range(period, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    tr_smooth = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(tr_smooth == 0, np.nan, tr_smooth)
    di_minus = 100 * dm_minus_smooth / np.where(tr_smooth == 0, np.nan, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, np.nan, (di_plus + di_minus))
    adx_14 = np.zeros_like(dx)
    adx_14[27:] = np.nanmean(dx[14:28])  # seed ADX with first 14-period average of DX
    for i in range(28, len(dx)):
        adx_14[i] = (1/14) * dx[i] + (13/14) * adx_14[i-1]
    
    # Align 1d ADX to 4h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Bollinger Bands on 4h data (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_bb + (bb_std_dev * bb_std)
    lower_band = sma_bb - (bb_std_dev * bb_std)
    middle_band = sma_bb  # 20-period SMA
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_14_aligned[i]) or np.isnan(sma_bb[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_14_aligned[i]
        sma_val = sma_bb[i]
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above upper BB AND ADX > 25 (trending) AND volume spike
            if (price > upper_val and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB AND ADX > 25 (trending) AND volume spike
            elif (price < lower_val and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle BB OR volume drops below average
                if (price <= sma_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle BB OR volume drops below average
                if (price >= sma_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BB_Squeeze_ADX25_Volume"
timeframe = "4h"
leverage = 1.0