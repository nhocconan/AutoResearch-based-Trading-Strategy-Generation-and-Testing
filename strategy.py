#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Fisher Transform + 1d ADX regime + volume confirmation
# Fisher Transform catches reversals in ranging markets (common in 2025 bear/range)
# 1d ADX > 25 filters for trending regimes to avoid whipsaws
# Volume confirmation ensures conviction on signals
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_FisherTransform_1dADX_VolumeConfirm"
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
    
    # Volume confirmation: 24-period MA (equivalent to 1d lookback in 6h)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d timeframe (standard 14-period)
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # Align with original arrays
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            if len(data) < period:
                return np.full(len(data), np.nan)
            result = np.full(len(data), np.nan)
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smoothing(tr, 14)
        plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(len(close_1d), np.nan)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Fisher Transform on 6h timeframe (period=10)
    if len(close) >= 10:
        # Median price
        median_price = (high + low) / 2
        
        # Normalize median price to [-1, 1] range over lookback period
        def normalize(series, period):
            if len(series) < period:
                return np.full(len(series), np.nan)
            result = np.full(len(series), np.nan)
            for i in range(period-1, len(series)):
                min_val = np.nanmin(series[i-period+1:i+1])
                max_val = np.nanmax(series[i-period+1:i+1])
                if max_val - min_val != 0:
                    result[i] = 2 * ((series[i] - min_val) / (max_val - min_val)) - 1
                else:
                    result[i] = 0
            return result
        
        normalized = normalize(median_price, 10)
        
        # Avoid extreme values for ln calculation
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher Transform formula: 0.5 * ln((1+X)/(1-X))
        fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Signal line: 3-period EMA of Fisher
        if len(fisher) >= 3:
            fisher_signal = pd.Series(fisher).ewm(span=3, adjust=False, min_periods=3).mean().values
        else:
            fisher_signal = np.full(len(fisher), np.nan)
    else:
        fisher = np.full(n, np.nan)
        fisher_signal = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(fisher[i]) or np.isnan(fisher_signal[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above signal line + ADX > 25 (trending) + volume filter
            if (fisher[i] > fisher_signal[i] and 
                fisher[i-1] <= fisher_signal[i-1] and  # Cross above
                adx_aligned[i] > 25 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Fisher crosses below signal line + ADX > 25 (trending) + volume filter
            elif (fisher[i] < fisher_signal[i] and 
                  fisher[i-1] >= fisher_signal[i-1] and  # Cross below
                  adx_aligned[i] > 25 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Fisher crosses below signal line OR ADX drops below 20 (losing trend)
            if (fisher[i] < fisher_signal[i] and 
                fisher[i-1] >= fisher_signal[i-1]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Fisher crosses above signal line OR ADX drops below 20
            if (fisher[i] > fisher_signal[i] and 
                fisher[i-1] <= fisher_signal[i-1]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals