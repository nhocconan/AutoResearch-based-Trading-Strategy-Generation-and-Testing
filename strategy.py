#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume spike confirmation
# Long when: Bull Power > 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Short when: Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA
# Exit when: Elder Power reverses sign OR ADX < 20 (range) OR volume normalizes
# Uses Elder Ray for trend strength, 1d ADX for regime filter, volume spike for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_ElderRay_1dADX_VolumeConfirm"
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
    
    # Calculate Elder Ray on 6h (13-period EMA for reference)
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
        bull_power = high - ema_13
        bear_power = low - ema_13
    else:
        bull_power = np.full(n, np.nan)
        bear_power = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d timeframe
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) >= period:
                # First value is simple average
                result[period-1] = np.nanmean(data[:period])
                # Subsequent values: Wilder's smoothing
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]):
                        result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilder_smooth(tr, 14)
        plus_dm_smooth = wilder_smooth(plus_dm, 14)
        minus_dm_smooth = wilder_smooth(minus_dm, 14)
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * (plus_dm_smooth / atr), 0)
        minus_di = np.where(atr != 0, 100 * (minus_dm_smooth / atr), 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smooth(dx, 14)
        
        # ADX regime: >25 = trending, <20 = ranging
        adx_trending = adx > 25
        adx_ranging = adx < 20
    else:
        adx = np.full(len(close_1d), np.nan)
        adx_trending = np.full(len(close_1d), False)
        adx_ranging = np.full(len(close_1d), False)
    
    # Align 1d ADX regime to 6h timeframe
    adx_trending_aligned = align_htf_to_ltf(prices, df_1d, adx_trending.astype(float))
    adx_ranging_aligned = align_htf_to_ltf(prices, df_1d, adx_ranging.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_trending_aligned[i]) or 
            np.isnan(adx_ranging_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND 1d ADX trending AND volume filter
            if (bull_power[i] > 0 and 
                adx_trending_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND 1d ADX trending AND volume filter
            elif (bear_power[i] < 0 and 
                  adx_trending_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR ADX ranging OR volume normalizes
            if (bull_power[i] <= 0 or adx_ranging_aligned[i] == 1.0 or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR ADX ranging OR volume normalizes
            if (bear_power[i] >= 0 or adx_ranging_aligned[i] == 1.0 or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals