#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX trend strength + 12h Donchian(20) breakout direction + volume confirmation
# ADX > 25 indicates strong trend (works in bull/bear markets)
# 12h Donchian breakout provides directional bias with fewer false signals
# 6h volume spike confirms breakout authenticity
# Discrete sizing 0.25 to manage drawdown in volatile 6h timeframe
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_12h_adx_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX and Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing for TR, DM+, DM-
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smoothed / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smoothed / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h indicators to 6h timeframe (wait for 12h bar close)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    highest_high_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    lowest_low_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    
    # Calculate 6h average volume (20-period) for confirmation
    volume_s = pd.Series(volume)
    avg_volume_6h = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high_12h_aligned[i]) or 
            np.isnan(lowest_low_12h_aligned[i]) or np.isnan(avg_volume_6h[i])):
            signals[i] = 0.0
            continue
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current 6h volume > 1.8x 6h average volume
        volume_confirmed = volume[i] > 1.8 * avg_volume_6h[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian lower band OR trend weakens
            if close[i] < lowest_low_12h_aligned[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian upper band OR trend weakens
            if close[i] > highest_high_12h_aligned[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: only in strong trend with volume confirmation
            if strong_trend and volume_confirmed:
                if close[i] > highest_high_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lowest_low_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals