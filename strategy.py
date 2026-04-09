#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal + 1d ADX trend filter + volume spike confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# 1d ADX > 25 ensures we only trade in trending markets (avoids choppy whipsaws)
# Volume spike (>2x average) confirms institutional participation at reversal points
# Works in bull/bear: ADX filters for trending environments, Williams %R captures pullbacks within trend
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_williamsr_adx_volume_v2"
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
    
    # Load 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period ADX for 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with 1d index
    
    # Directional Movement
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe (wait for daily close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R (14-period)
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if i < 13:
            williams_r[i] = np.nan
        else:
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high == lowest_low:
                williams_r[i] = -50.0  # Avoid division by zero
            else:
                williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
    
    # Calculate 20-period average volume for volume spike confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR ADX < 20 (trend weakening)
            if williams_r[i] > -20 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR ADX < 20 (trend weakening)
            if williams_r[i] < -80 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, Williams %R extremes + ADX filter
            if volume_confirmed and adx_aligned[i] > 25:
                # Long entry: Williams %R < -80 (oversold) in uptrend
                if williams_r[i] < -80:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Williams %R > -20 (overbought) in uptrend
                elif williams_r[i] > -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals