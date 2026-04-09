#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter
# Williams %R identifies overbought/oversold conditions on 6h chart
# 1d ADX > 25 filters for trending markets to avoid whipsaws in ranging markets
# Long: Williams %R < -80 (oversold) + ADX > 25 (strong trend)
# Short: Williams %R > -20 (overbought) + ADX > 25 (strong trend)
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: trend filter ensures we only trade with momentum

name = "6h_1d_williamsr_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing function
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # Smoothed TR, +DM, -DM
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 14)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_1d = np.where(atr_1d > 0, 100 * plus_dm_smoothed / atr_1d, 0.0)
    minus_di_1d = np.where(atr_1d > 0, 100 * minus_dm_smoothed / atr_1d, 0.0)
    
    # DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) > 0, 
                     100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d), 
                     0.0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 6h Williams %R(14)
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = np.where((highest_high_6h - lowest_low_6h) != 0,
                             -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h),
                             -50)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_6h[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long when Williams %R crosses above -50 (exiting oversold)
            if williams_r_6h[i] > -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short when Williams %R crosses below -50 (exiting overbought)
            if williams_r_6h[i] < -50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when Williams %R < -80 (oversold) AND ADX > 25 (strong trend)
            if williams_r_6h[i] < -80 and adx_1d_aligned[i] > 25:
                position = 1
                signals[i] = 0.25
            # Enter short when Williams %R > -20 (overbought) AND ADX > 25 (strong trend)
            elif williams_r_6h[i] > -20 and adx_1d_aligned[i] > 25:
                position = -1
                signals[i] = -0.25
    
    return signals