#!/usr/bin/env python3
"""
4h_ADX_Donchian_Breakout_Volume
4h strategy combining Donchian breakout with ADX trend strength and volume confirmation.
Long: Price breaks above Donchian(20) high + ADX > 25 + volume > 1.5x average
Short: Price breaks below Donchian(20) low + ADX > 25 + volume > 1.5x average
Exit: Opposite breakout or ADX falling below 20
Uses 12h trend filter via EMA crossover to avoid counter-trend trades
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in trending markets (breakouts) and avoids choppy periods via ADX filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- (14-period Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    
    # 12h EMA trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough for ADX and EMAs
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx[i]) or np.isnan(ema_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend_12h = ema_20_aligned[i] > ema_50_aligned[i]
        downtrend_12h = ema_20_aligned[i] < ema_50_aligned[i]
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > high_20[i]
        breakdown_down = close[i] < low_20[i]
        
        if position == 0:
            # Long: uptrend + strong trend + volume + breakout above Donchian high
            if uptrend_12h and strong_trend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + strong trend + volume + breakdown below Donchian low
            elif downtrend_12h and strong_trend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakness, opposite breakout, or trend reversal
            if weak_trend or breakdown_down or not uptrend_12h:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakness, opposite breakout, or trend reversal
            if weak_trend or breakout_up or not downtrend_12h:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0