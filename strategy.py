#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 Breakout + 1d Volume Spike + ADX Regime Filter
# Camarilla R3/S3 levels act as strong intraday support/resistance. Breakouts with volume confirmation capture momentum.
# 1d volume spike filter ensures institutional participation. ADX > 25 filters for trending markets, avoiding chop.
# Designed for 20-50 trades/year on 4h to minimize fee drag while capturing strong moves in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume EMA (20-period)
    vol_ema_20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ema_20_1d * 2.0)  # Volume at least 2x average
    
    # Align 1d volume spike to 4h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate ADX (14-period) on 4h for regime filter
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ADX > 25 indicates trending market
    trending = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(volume_spike_aligned[i]) or 
            np.isnan(trending[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0 and trending[i]:
            # Calculate Camarilla levels for today (using previous day's OHLC)
            # Need to get previous day's data - we'll approximate using rolling window
            if i >= 24:  # Need at least 24 hours of data (6 4h bars) for previous day
                # Get high, low, close of previous day (24 hours ago)
                prev_high = np.max(high[i-24:i]) if i >= 24 else high[i]
                prev_low = np.min(low[i-24:i]) if i >= 24 else low[i]
                prev_close = close[i-1] if i >= 1 else close[i]
                
                # Camarilla levels
                range_val = prev_high - prev_low
                if range_val > 0:
                    camarilla_r3 = prev_close + (range_val * 1.1 / 4)
                    camarilla_s3 = prev_close - (range_val * 1.1 / 4)
                    
                    # Long conditions: price breaks above R3 with volume spike
                    if (close[i] > camarilla_r3 and 
                        volume_spike_aligned[i]):
                        signals[i] = 0.25
                        position = 1
                    # Short conditions: price breaks below S3 with volume spike
                    elif (close[i] < camarilla_s3 and 
                          volume_spike_aligned[i]):
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: price falls below S3 or loses volume spike
            if (close[i] < camarilla_s3 or 
                not volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 or loses volume spike
            if (close[i] > camarilla_r3 or 
                not volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals