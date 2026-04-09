#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with volume confirmation and ADX regime filter
# Williams %R > -20 = overbought (short signal), < -80 = oversold (long signal)
# Only trade in strong trends (ADX > 25) to avoid whipsaws in ranging markets
# Volume confirmation ensures breakouts have conviction
# Designed for both bull and bear markets: rides trends while filtering countertrend noise
# Target: 12-37 trades/year (50-150 over 4 years) with discrete sizing 0.25 to minimize fee drag

name = "6h_1d_williamsr_volume_adx_v2"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = np.where(
        (highest_high_1d - lowest_low_1d) != 0,
        ((highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d)) * -100,
        -50  # neutral when range=0
    )
    
    # Calculate 1d ADX (14-period) for trend strength
    # ADX calculation requires +DI and -DI
    def calculate_dx(high, low, close, period):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        def wilders_smoothing(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # +DI and -DI
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        # DX
        dx = np.where((plus_di + minus_di) != 0, 
                      100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                      0)
        
        # ADX = smoothed DX
        adx = wilders_smoothing(dx, period)
        return adx, plus_di, minus_di
    
    adx_1d, _, _ = calculate_dx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_1d_aligned[i] <= 25:
            # In ranging markets, stay flat or mean revert minimally
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -20 (overbought) or trend weakens
            if williams_r_1d_aligned[i] > -20 or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -80 (oversold) or trend weakens
            if williams_r_1d_aligned[i] < -80 or adx_1d_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when oversold (-80 or below) with volume confirmation
            if williams_r_1d_aligned[i] <= -80 and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short when overbought (-20 or above) with volume confirmation
            elif williams_r_1d_aligned[i] >= -20 and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals