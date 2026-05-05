#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) + 12h ADX trend filter + volume confirmation
# Long when: Bull Power > 0 AND ADX(12h) > 25 AND volume > 1.5x 20-period MA
# Short when: Bear Power < 0 AND ADX(12h) > 25 AND volume > 1.5x 20-period MA
# Exit when: Bull/Bear Power crosses zero OR ADX < 20 (trend weakening)
# Uses Elder Ray for trend strength measurement, ADX for regime filter, volume for conviction
# Timeframe: 6h, HTF: 12h. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_12hADX_VolumeConfirm"
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
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Elder Ray Index on 6h (13-period EMA)
    if len(high) >= 13 and len(low) >= 13 and len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high - ema_13
        bear_power = low - ema_13
    else:
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # Get 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components on 12h timeframe
    if len(high_12h) >= 14 and len(low_12h) >= 14 and len(close_12h) >= 14:
        # True Range
        tr1 = high_12h[1:] - low_12h[1:]
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with original index
        
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilder_smooth(data, period):
            if len(data) < period:
                return np.full_like(data, np.nan)
            result = np.full_like(data, np.nan)
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        period = 14
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / atr
        minus_di = 100 * wilder_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
        
        # ADX trend filter
        adx_strong = adx > 25
        adx_weak = adx < 20
    else:
        adx_strong = np.full(len(close_12h), False)
        adx_weak = np.full(len(close_12h), False)
    
    # Align 12h ADX to 6h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_12h, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_12h, adx_weak.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_strong_aligned[i]) or np.isnan(adx_weak_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND ADX strong AND volume filter
            if (bull_power[i] > 0 and 
                adx_strong_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND ADX strong AND volume filter
            elif (bear_power[i] < 0 and 
                  adx_strong_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR ADX weak (trend weakening)
            if (bull_power[i] <= 0 or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR ADX weak (trend weakening)
            if (bear_power[i] >= 0 or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals