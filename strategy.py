#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
    # Donchian breakouts capture momentum, EMA50 filters trend direction, volume confirms strength
    # Works in bull markets (breakouts continue) and bear markets (fades false breakouts)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channels (20-period) on 1d
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    upper = rolling_max(high_1d, 20)
    lower = rolling_min(low_1d, 20)
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation (20-period average on 1d)
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    vol_spike = volume_1d > 1.5 * vol_ma20_aligned  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + above 1w EMA50 + volume spike
            if close_1d[i] > upper[i] and close_1d[i] > ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + below 1w EMA50 + volume spike
            elif close_1d[i] < lower[i] and close_1d[i] < ema50_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Donchian channel or trend reversal
            middle = (upper[i] + lower[i]) / 2
            if position == 1:
                if close_1d[i] < middle or close_1d[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_1d[i] > middle or close_1d[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0