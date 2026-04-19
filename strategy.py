#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index + volume confirmation + 1d ADX trend filter
# Elder Ray consists of Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Strong uptrend: Bull Power > 0 and rising, Bear Power < 0
# Strong downtrend: Bear Power < 0 and falling, Bull Power > 0
# Combined with volume confirmation and daily ADX trend filter to reduce false signals
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_ElderRay_1dADX_Volume"
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
    
    # Daily ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed values
    def smoothed_average(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = smoothed_average(tr, 14)
    dm_plus_smoothed = smoothed_average(dm_plus, 14)
    dm_minus_smoothed = smoothed_average(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smoothed_average(dx, 14)
    adx_1d = adx  # Already calculated on daily data
    
    # Align daily ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Elder Ray components on 4h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when ADX > 25 (trending market)
        if adx_1d_aligned[i] > 25:
            if position == 0:
                # Long: Bull Power > 0 and rising, Bear Power < 0, with volume spike
                if (bull_power[i] > 0 and i > start_idx and bull_power[i] > bull_power[i-1] and 
                    bear_power[i] < 0 and volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 and falling, Bull Power > 0, with volume spike
                elif (bear_power[i] < 0 and i > start_idx and bear_power[i] < bear_power[i-1] and 
                      bull_power[i] > 0 and volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                    
            elif position == 1:
                # Long: exit if Bull Power turns negative or ADX weakens
                if bull_power[i] <= 0 or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
            elif position == -1:
                # Short: exit if Bear Power turns positive or ADX weakens
                if bear_power[i] >= 0 or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In ranging markets, stay flat
            signals[i] = 0.0
            position = 0
    
    return signals