#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 12h volume confirmation and chop regime filter
    # Donchian(20) breakout captures strong momentum moves
    # 12h volume spike (>2.0x 24-period average) confirms institutional participation
    # Chop regime filter (CHOP > 61.8 = range, < 38.2 = trend) avoids whipsaws
    # Designed for low trade frequency (target: 20-40/year) to minimize fee drag
    # Works in bull/bear markets by only trading strong volume-confirmed breakouts in trending regimes
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels: upper = 20-period high, lower = 20-period low
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper = rolling_max(high_4h, 20)
    donchian_lower = rolling_min(low_4h, 20)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 24:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(df_12h), np.nan)
    for i in range(24, len(df_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-24:i])
    
    # Volume confirmation: volume > 2.0 * 24-period average (12h)
    volume_spike_12h = volume_12h > (2.0 * vol_ma_12h)
    
    # Get 4h data for chop regime filter (using 4h for higher resolution)
    # Chop = 100 * log10(sum(ATR(1),14) / (max(high,14) - min(low,14))) / log10(14)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    if len(df_4h) >= 14:
        tr = np.full(len(df_4h), np.nan)
        for i in range(1, len(df_4h)):
            tr[i] = true_range(high_4h[i], low_4h[i], close_4h[i-1])
        
        atr_1 = np.full(len(df_4h), np.nan)
        for i in range(len(df_4h)):
            if i == 0:
                atr_1[i] = tr[i]
            else:
                atr_1[i] = (atr_1[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
        
        sum_atr_14 = np.full(len(df_4h), np.nan)
        for i in range(13, len(df_4h)):
            sum_atr_14[i] = np.sum(atr_1[i-13:i+1])
        
        max_high_14 = rolling_max(high_4h, 14)
        min_low_14 = rolling_min(low_4h, 14)
        
        chop_denom = max_high_14 - min_low_14
        chop = np.full(len(df_4h), np.nan)
        for i in range(len(df_4h)):
            if chop_denom[i] > 0 and not np.isnan(sum_atr_14[i]):
                chop[i] = 100 * np.log10(sum_atr_14[i] / chop_denom[i]) / np.log10(14)
        
        # Chop regime: > 61.8 = range, < 38.2 = trend
        chop_trend = chop < 38.2  # Only trade in trending regimes
    else:
        chop_trend = np.full(len(df_4h), True)  # Default to trend if not enough data
    
    # Align all indicators to LTF (4h is our primary timeframe, so minimal alignment needed)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    chop_trend_aligned = align_htf_to_ltf(prices, df_4h, chop_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_upper_aligned[i]
        bearish_breakout = close[i] < donchian_lower_aligned[i]
        
        # Entry logic: Donchian breakout + volume confirmation + chop trend filter
        long_entry = False
        short_entry = False
        
        # Long: bullish breakout + volume spike + trending regime
        if bullish_breakout:
            long_entry = volume_spike_aligned[i] and chop_trend_aligned[i]
        # Short: bearish breakout + volume spike + trending regime
        elif bearish_breakout:
            short_entry = volume_spike_aligned[i] and chop_trend_aligned[i]
        
        # Exit logic: opposite breakout or loss of volume confirmation
        long_exit = bearish_breakout or not volume_spike_aligned[i]
        short_exit = bullish_breakout or not volume_spike_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0