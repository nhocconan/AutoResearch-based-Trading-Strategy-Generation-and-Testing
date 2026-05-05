#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d volatility regime filter and volume confirmation
# Long when price breaks above 12h Donchian(20) upper band AND 1d ATR ratio < 0.8 (low volatility regime) AND volume > 1.5 * avg_volume(20) on 6h
# Short when price breaks below 12h Donchian(20) lower band AND 1d ATR ratio < 0.8 (low volatility regime) AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price crosses back through the 12h Donchian midpoint
# Uses discrete sizing 0.25 to balance return and risk
# Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe
# 12h Donchian provides stable structure with fewer whipsaws than shorter timeframes
# 1d ATR ratio (ATR(7)/ATR(30)) < 0.8 identifies low volatility regimes where breakouts tend to persist
# Volume confirmation (1.5x) validates breakout strength while avoiding overtrading

name = "6h_12hDonchian20_1dATRRegime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need at least 20 completed 12h bars for Donchian
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper_12h = rolling_max(high_12h, 20)
    donchian_lower_12h = rolling_min(low_12h, 20)
    donchian_mid_12h = (donchian_upper_12h + donchian_lower_12h) / 2.0
    
    # Align 12h Donchian to 6h timeframe (wait for completed 12h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_12h)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 completed daily bars for ATR(30)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate ATR(7) and ATR(30) using Wilder's smoothing
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average (skip first NaN)
        result[period] = np.nansum(data[1:period+1]) / period
        # Wilder's smoothing: previous * (period-1)/period + current/period
        for i in range(period+1, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (period-1)/period + data[i]/period
        return result
    
    atr_7 = wilders_smoothing(tr, 7)
    atr_30 = wilders_smoothing(tr, 30)
    
    # ATR ratio: ATR(7) / ATR(30) < 0.8 indicates low volatility regime
    atr_ratio = np.where(atr_30 != 0, atr_7 / atr_30, np.nan)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper, ATR ratio < 0.8 (low vol), volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                atr_ratio_aligned[i] < 0.8 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower, ATR ratio < 0.8 (low vol), volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  atr_ratio_aligned[i] < 0.8 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals