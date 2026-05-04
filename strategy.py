#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation (2x vol EMA20)
# Donchian breakout captures momentum, HMA filters trend direction, volume confirms strength
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)
# Discrete sizing 0.25 targets 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Donchian20_HMA21_VolumeConfirm"
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
    
    # Get 1d data for HTF context (optional, can remove if not needed)
    # df_1d = get_htf_data(prices, '1d')
    # if len(df_1d) < 34:
    #     return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate HMA(21) for trend filter
    def hull_moving_average(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA for half period
        weights = np.arange(1, half_period + 1)
        wma_half = np.convolve(arr, weights / weights.sum(), mode='same')
        wma_half[:half_period-1] = np.nan
        wma_half[-half_period+1:] = np.nan
        
        # WMA for full period
        weights_full = np.arange(1, period + 1)
        wma_full = np.convolve(arr, weights_full / weights_full.sum(), mode='same')
        wma_full[:period-1] = np.nan
        wma_full[-period+1:] = np.nan
        
        # WMA for sqrt period
        weights_sqrt = np.arange(1, sqrt_period + 1)
        wma_sqrt = np.convolve(2 * wma_half - wma_full, weights_sqrt / weights_sqrt.sum(), mode='same')
        wma_sqrt[:sqrt_period-1] = np.nan
        wma_sqrt[-sqrt_period+1:] = np.nan
        
        return wma_sqrt
    
    hma_21 = hull_moving_average(close, 21)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND HMA uptrend AND volume spike
            if close[i] > highest_high[i] and close[i] > hma_21[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND HMA downtrend AND volume spike
            elif close[i] < lowest_low[i] and close[i] < hma_21[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR HMA turns down
            if close[i] < lowest_low[i] or close[i] < hma_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR HMA turns up
            if close[i] > highest_high[i] or close[i] > hma_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals