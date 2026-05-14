#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h ADX trend filter and volume spike confirmation.
# Enter long when price breaks above Camarilla R4 level with 12h ADX > 25 (trending) and volume > 1.5x 20-bar average.
# Enter short when price breaks below Camarilla S4 level with 12h ADX > 25 and volume confirmation.
# Exit when price retraces to the Camarilla H4/L4 levels respectively.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla R4/S4 levels are stronger breakout points than R3/S3, reducing false breakouts.
# 12h ADX ensures we only trade in trending markets, avoiding choppy conditions.
# Volume spike filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "4h_Camarilla_R4S4_Breakout_12hADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])  # Skip first NaN
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 4h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels (R4/S4 are stronger breakout points)
    R4 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 2
    S4 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 2
    H4 = prev_close_aligned + (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    L4 = prev_close_aligned - (prev_high_aligned - prev_low_aligned) * 1.1 / 4
    
    # Volume confirmation: >1.5x 20-bar average volume (less strict than 2x to allow more trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure sufficient history for volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i]) or np.isnan(R4[i]) or np.isnan(S4[i]) or
            np.isnan(H4[i]) or np.isnan(L4[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 12h ADX trend: ADX > 25 indicates trending market
        adx_trending = adx_aligned[i] > 25
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R4, ADX trending, volume confirm
            if price > R4[i] and adx_trending and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < S4, ADX trending, volume confirm
            elif price < S4[i] and adx_trending and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H4
            if price <= H4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at L4
            if price >= L4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals