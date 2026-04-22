#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray Power (Bull/Bear) + 1d ADX25 regime filter + volume confirmation
    # Works in both bull and bear: Elder Ray captures institutional buying/selling pressure
    # ADX25 filters for trending regimes (avoids chop), volume confirms institutional participation
    # Bull Power > 0 + Bear Power < 0 indicates strong trend with continuation
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ADX (14) for trend strength
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[1:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    atr = wilder_smooth(tr, 14)
    di_plus = 100 * wilder_smooth(dm_plus, 14) / atr
    di_minus = 100 * wilder_smooth(dm_minus, 14) / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, 14)
    adx_14 = adx
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Daily Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13  # Bull Power = High - EMA13
    bear_power = low_1d - ema13   # Bear Power = Low - EMA13
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter (20-period surge)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(adx_14_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power + weak bear power + ADX > 25 (trending) + volume surge
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and adx_14_aligned[i] > 25 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power + weak bull power + ADX > 25 (trending) + volume surge
            elif bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and adx_14_aligned[i] > 25 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: ADX drops below 20 (trend weakening) OR power signals weaken
            if position == 1:
                if adx_14_aligned[i] < 20 or bull_power_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if adx_14_aligned[i] < 20 or bear_power_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_ADX25_Trend_VolumeSurge_v1"
timeframe = "6h"
leverage = 1.0