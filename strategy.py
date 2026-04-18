#!/usr/bin/env python3
"""
6h ADX + Williams Alligator + Volume Spike
Trend-following strategy using Williams Alligator (3 SMAs) and ADX for trend confirmation.
Long when price > Alligator teeth and ADX > 25, short when price < Alligator teeth and ADX > 25.
Uses volume spike (2x 6-period average) for entry confirmation.
Designed for 6h timeframe with daily trend filter to avoid false signals in choppy markets.
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
    
    # Williams Alligator: Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            result[period-1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close, 13)   # Jaw (blue)
    teeth = smma(close, 8)   # Teeth (red)
    lips = smma(close, 5)    # Lips (green)
    
    # ADX calculation for trend strength
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smm(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) >= period:
                result[period-1] = np.nansum(arr[:period])
                for i in range(period, len(arr)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] - (result[i-1] / period) + arr[i]
            return result
        
        tr_smooth = smm(tr, period)
        plus_dm_smooth = smm(plus_dm, period)
        minus_dm_smooth = smm(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = smm(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume spike detection (2x 6-period average)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get daily trend filter (1 = uptrend, -1 = downtrend, 0 = neutral)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Daily EMA 34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend = np.where(close_1d > ema_34_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i]) or np.isnan(daily_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        strong_trend = adx[i] > 25
        price_above_teeth = price > teeth[i]
        price_below_teeth = price < teeth[i]
        
        if position == 0:
            # Long: price above teeth, strong trend, volume spike, daily uptrend
            if (price_above_teeth and strong_trend and volume_spike[i] and 
                daily_trend_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: price below teeth, strong trend, volume spike, daily downtrend
            elif (price_below_teeth and strong_trend and volume_spike[i] and 
                  daily_trend_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below teeth or trend weakens
            if price < teeth[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above teeth or trend weakens
            if price > teeth[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_VolumeSpike_DailyTrend"
timeframe = "6h"
leverage = 1.0