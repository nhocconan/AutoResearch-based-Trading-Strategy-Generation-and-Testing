#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with daily volume confirmation and weekly ADX trend filter.
# Long when price breaks above weekly Donchian high (20-period) with daily volume > 1.5x 20-day EMA and weekly ADX > 25.
# Short when price breaks below weekly Donchian low (20-period) with same volume and ADX conditions.
# Exit when price crosses back inside the weekly Donchian channel or ADX falls below 20.
# Designed for low trade frequency (<10/year) to avoid fee drag. Works in trending markets with volatility filter.

name = "1d_1wDonchian_Breakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high_1w, 20)
    donchian_low = rolling_min(low_1w, 20)
    
    # Calculate weekly ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        def Wilder_smooth(data, period):
            smoothed = np.full_like(data, np.nan)
            if len(data) < period:
                return smoothed
            smoothed[period-1] = np.nanmean(data[1:period])
            for i in range(period, len(data)):
                smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            return smoothed
        
        atr = Wilder_smooth(tr, period)
        dmp = Wilder_smooth(dm_plus, period)
        dmm = Wilder_smooth(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, 100 * dmp / atr, 0)
        di_minus = np.where(atr != 0, 100 * dmm / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = Wilder_smooth(dx, period)
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily volume confirmation: volume > 1.5x 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high with volume and ADX > 25
            if (close[i] > donchian_high_aligned[i] and 
                vol_confirm[i] and 
                adx_1w_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low with volume and ADX > 25
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_confirm[i] and 
                  adx_1w_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back inside weekly Donchian channel or ADX < 20
            if (close[i] < donchian_high_aligned[i] or 
                adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back inside weekly Donchian channel or ADX < 20
            if (close[i] > donchian_low_aligned[i] or 
                adx_1w_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals