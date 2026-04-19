#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) with 1d ADX(14) trend filter and volume confirmation.
# Long when Williams %R crosses above -20 from below, ADX > 25, and volume > 1.5x 4h average volume.
# Short when Williams %R crosses below -80 from above, ADX > 25, and volume > 1.5x 4h average volume.
# Exit when Williams %R crosses back below -50 (long) or above -50 (short).
# Williams %R identifies overbought/oversold conditions, ADX filters for trending markets,
# volume confirms momentum. Target: 20-50 trades/year per symbol.
name = "4h_WilliamsR_ADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    # Get daily data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30)  # Ensure Williams %R and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(willr[i]) or np.isnan(willr[i-1]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        willr_now = willr[i]
        willr_prev = willr[i-1]
        adx = adx_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: Williams %R crosses above -20 from below, ADX > 25, volume confirmation
            if willr_prev <= -20 and willr_now > -20 and adx > 25 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -80 from above, ADX > 25, volume confirmation
            elif willr_prev >= -80 and willr_now < -80 and adx > 25 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses below -50 from above
            if willr_prev >= -50 and willr_now < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses above -50 from below
            if willr_prev <= -50 and willr_now > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals