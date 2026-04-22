#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. ADX filters for trending markets.
# Long: %R < -80 (oversold) + ADX > 25 (trending) + volume > 1.5x average
# Short: %R > -20 (overbought) + ADX > 25 (trending) + volume > 1.5x average
# Works in both bull and bear markets by only taking trades in the direction of the trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Plus Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_plus[0] = 0
        
        # Minus Directional Movement
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_minus[0] = 0
        
        # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                # Wilder smoothing: today = (1-alpha)*yesterday + alpha*today
                for i in range(period, len(data)):
                    result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # Avoid division by zero
        plus_di = 100 * dm_plus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
        minus_di = 100 * dm_minus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Williams %R(14) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        # For periods before we have enough data, use expanding window
        for i in range(period-1):
            highest_high[i] = np.max(high[:i+1])
            lowest_low[i] = np.min(low[:i+1])
        wr = -100 * (highest_high - close) / np.where((highest_high - lowest_low) == 0, 1, (highest_high - lowest_low))
        return wr
    
    williams_r_14 = williams_r(high, low, close, 14)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    # For periods before we have enough data, use expanding window average
    for i in range(20):
        vol_avg_20[i] = np.mean(volume[:i+1]) if i+1 > 0 else 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after ADX and Williams %R warmup
        # Skip if data not ready
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(williams_r_14[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) + ADX > 25 (trending) + volume spike
            if (williams_r_14[i] < -80 and 
                adx_14_1d_aligned[i] > 25 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) + ADX > 25 (trending) + volume spike
            elif (williams_r_14[i] > -20 and 
                  adx_14_1d_aligned[i] > 25 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R reverses or ADX weakens
            if position == 1:
                # Exit long: Williams %R >= -20 (overbought) or ADX < 20 (weak trend)
                if (williams_r_14[i] >= -20 or 
                    adx_14_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Williams %R <= -80 (oversold) or ADX < 20 (weak trend)
                if (williams_r_14[i] <= -80 or 
                    adx_14_1d_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dADX25_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0