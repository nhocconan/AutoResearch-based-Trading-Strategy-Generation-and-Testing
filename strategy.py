#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
# Uses 1d ADX > 25 to confirm strong trending markets (avoids chop/range)
# Donchian breakout captures momentum in direction of 1d trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in both bull and bear markets by following 1d ADX trend direction
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag
# Discrete position sizing: 0.25 (25% of capital) to minimize fee churn while maintaining reasonable exposure

name = "6h_Donchian20_Breakout_1dADX_Volume"
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
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian upper/lower bands (20-period)
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX(14) for trend strength filter
    # ADX requires +DI, -DI, and TR calculation
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (similar to EMA with alpha=1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: smoothed = prev_smoothed * (1 - 1/period) + current * (1/period)
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align HTF indicators to 6h timeframe
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 34, 20)  # Donchian(20), ADX(14+14 for smoothing), volume MA(20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND ADX > 25 (strong trend) AND volume spike
            if (close[i] > donch_high_1d_aligned[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND ADX > 25 (strong trend) AND volume spike
            elif (close[i] < donch_low_1d_aligned[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR ADX falls below 20 (trend weakening)
            if close[i] < donch_low_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR ADX falls below 20 (trend weakening)
            if close[i] > donch_high_1d_aligned[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals