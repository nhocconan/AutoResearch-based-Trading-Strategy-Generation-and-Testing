#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6-hour trend following with 1-day ADX filter and volume spike confirmation
    # Uses 6h EMA crossover for entry, filtered by 1d ADX>25 (trending market) and volume spike
    # Designed to work in both bull and bear markets by only trading in strong trends
    # Target: 15-25 trades/year to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on daily data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First value has no previous close
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = np.nan
        down_move[0] = np.nan
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(arr[:period]) / period
            # Subsequent values are Wilder smoothing
            for i in range(period, len(arr)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smooth_wilder(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h EMA crossover (fast=9, slow=21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA9 crosses above EMA21 with ADX>25 and volume spike
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and adx_1d_aligned[i] > 25 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: EMA9 crosses below EMA21 with ADX>25 and volume spike
            elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and adx_1d_aligned[i] > 25 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: EMA crossover reverses OR ADX drops below 20 (trend weakening)
            if position == 1:
                if ema9[i] < ema21[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if ema9[i] > ema21[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ADX_Trend_EMA9_21_Volume_Spike_v1"
timeframe = "6h"
leverage = 1.0