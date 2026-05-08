#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and ADX trend filter.
# Donchian(20) breakouts capture momentum, volume confirms strength, ADX>25 ensures trending market.
# Long when price breaks above 12h Donchian high with volume and ADX>25.
# Short when price breaks below 12h Donchian low with volume and ADX>25.
# Designed for moderate trade frequency (20-40/year) to balance opportunity and cost.
# Works in both bull and bear markets by capturing directional breaks with trend confirmation.

name = "4h_DonchianBreakout_12hVolume_ADX"
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
    
    # Get 12h data for Donchian channels and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period)
    donch_high = np.zeros_like(close_12h)
    donch_low = np.zeros_like(close_12h)
    
    for i in range(len(close_12h)):
        start_idx = max(0, i - 19)
        donch_high[i] = np.max(high_12h[start_idx:i+1])
        donch_low[i] = np.min(low_12h[start_idx:i+1])
    
    # Calculate ADX (14-period) for trend strength
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])  # First TR is 0
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    tr14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX after 2*period
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Volume confirmation: current 12h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(df_12h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = df_12h['volume'].values > (vol_ema * 1.5)
    
    # Align indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above 12h Donchian high with volume and ADX>25
            if (close[i] > donch_high_aligned[i] and
                vol_confirm_aligned[i] > 0.5 and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: break below 12h Donchian low with volume and ADX>25
            elif (close[i] < donch_low_aligned[i] and
                  vol_confirm_aligned[i] > 0.5 and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below 12h Donchian low or ADX < 20
            if close[i] < donch_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 12h Donchian high or ADX < 20
            if close[i] > donch_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals