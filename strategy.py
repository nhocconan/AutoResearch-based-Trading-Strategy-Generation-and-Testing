#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with daily volume confirmation and weekly ADX trend filter
# Works in bull/bear: breakout captures momentum, volume confirms strength, ADX filters weak trends
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate daily ATR for volatility filter (14-period)
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate weekly ADX for trend filter (14-period)
    # Calculate True Range for weekly
    tr_1w = np.zeros(len(df_1w))
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    # Calculate Directional Movement
    plus_dm = np.zeros(len(df_1w))
    minus_dm = np.zeros(len(df_1w))
    for i in range(1, len(df_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(arr, period):
        smoothed = np.full_like(arr, np.nan)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    atr_1w_smooth = wilder_smooth(tr_1w, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Calculate DI+ and DI-
    plus_di_1w = np.full(len(df_1w), np.nan)
    minus_di_1w = np.full(len(df_1w), np.nan)
    dx_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        if atr_1w_smooth[i] > 0:
            plus_di_1w[i] = (plus_dm_smooth[i] / atr_1w_smooth[i]) * 100
            minus_di_1w[i] = (minus_dm_smooth[i] / atr_1w_smooth[i]) * 100
            di_sum = plus_di_1w[i] + minus_di_1w[i]
            if di_sum > 0:
                dx_1w[i] = (abs(plus_di_1w[i] - minus_di_1w[i]) / di_sum) * 100
    
    # Calculate ADX (smoothed DX)
    adx_1w = wilder_smooth(dx_1w, 14)
    adx_12h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate volume moving average (20-period) for daily volume
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        for i in range(19, len(df_1d)):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_ma_20_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12-hour Donchian channels (20-period) for entry signals
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h[i]) or
            np.isnan(adx_12h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_12h[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Require strong trend (ADX > 25)
        if adx_12h[i] < 25:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20_12h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_12h[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high with volume confirmation
            if close[i] > donch_high[i] and volume_ratio > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 12h Donchian low with volume confirmation
            elif close[i] < donch_low[i] and volume_ratio > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 12h Donchian low
            if close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 12h Donchian high
            if close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_Donchian_Volume_ADX"
timeframe = "12h"
leverage = 1.0