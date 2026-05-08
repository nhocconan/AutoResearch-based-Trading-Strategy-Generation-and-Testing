#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ADX trend filter + 4h Donchian breakout + volume confirmation
# Uses 1d ADX to filter trending markets (ADX > 25) and enters on 4h Donchian breakouts
# with volume surge confirmation. Designed for low trade frequency (<30/year) to avoid
# fee drag. Works in both bull and bear markets by only trading in strong trends.

name = "4h_ADXTrend_DonchianBreakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smooth TR and DM
        def smooth_wilder(arr, period):
            smoothed = np.full_like(arr, np.nan)
            smoothed[period] = np.nansum(arr[1:period+1])
            for i in range(period+1, len(arr)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
            return smoothed
        
        tr_smooth = smooth_wilder(tr, period)
        plus_dm_smooth = smooth_wilder(plus_dm, period)
        minus_dm_smooth = smooth_wilder(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = np.full_like(dx, np.nan)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h data for Donchian channel and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channel (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high_4h, 20)
    donchian_low = rolling_min(low_4h, 20)
    
    # Align Donchian levels to 4h timeframe (already 4h, but ensure proper alignment)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume spike: 2x 20-period EMA
    vol_ema = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume_4h > (vol_ema * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_4h, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when ADX indicates strong trend (> 25)
        if adx_1d_aligned[i] > 25:
            if position == 0:
                # Enter long: price breaks above Donchian high + volume surge
                if close[i] > donchian_high_aligned[i] and vol_spike_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price breaks below Donchian low + volume surge
                elif close[i] < donchian_low_aligned[i] and vol_spike_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price breaks below Donchian low
                if close[i] < donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above Donchian high
                if close[i] > donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # No trend - close any open positions
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals