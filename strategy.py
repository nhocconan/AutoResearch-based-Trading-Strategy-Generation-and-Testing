#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ADX_Trend_Filter_VolumeBreakout"
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
    
    # Calculate ADX for trend strength
    # TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # DM+
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_plus[0] = 0
    # DM-
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (1 - alpha) * atr[i-1] + alpha * tr[i]
    
    dm_plus_smooth = np.zeros(n)
    dm_minus_smooth = np.zeros(n)
    dm_plus_smooth[0] = dm_plus[0]
    dm_minus_smooth[0] = dm_minus[0]
    for i in range(1, n):
        dm_plus_smooth[i] = (1 - alpha) * dm_plus_smooth[i-1] + alpha * dm_plus[i]
        dm_minus_smooth[i] = (1 - alpha) * dm_minus_smooth[i-1] + alpha * dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros(n)
    adx[0] = dx[0]
    for i in range(1, n):
        adx[i] = (1 - alpha) * adx[i-1] + alpha * dx[i]
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend), price above daily EMA50, volume spike
            long_cond = (adx[i] > 25 and 
                        close[i] > ema_50_1d_aligned[i] and
                        volume_spike[i])
            
            # Short: ADX > 25 (strong trend), price below daily EMA50, volume spike
            short_cond = (adx[i] > 25 and 
                         close[i] < ema_50_1d_aligned[i] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: ADX drops below 20 (weakening trend) or price crosses below daily EMA50
            if adx[i] < 20 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX drops below 20 (weakening trend) or price crosses above daily EMA50
            if adx[i] < 20 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals