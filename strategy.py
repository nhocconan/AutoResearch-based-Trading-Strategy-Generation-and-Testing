#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Reversal with 1d ADX trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions - reversals from extreme levels often lead to sustained moves
# 1d ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranging conditions  
# Volume spike (>1.8 x 20-period EMA) confirms reversal validity with strong participation
# Discrete position sizing (0.25) controls fee drag while allowing meaningful exposure
# Target: 60-150 total trades over 4 years (15-38/year) for optimal risk-adjusted returns
# Works in bull markets by catching oversold bounces in uptrends, works in bear by selling overbought rallies in downtrends

name = "4h_WilliamsR_Reversal_1dADX_Trend_VolumeSpike"
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
    
    # Volume confirmation (volume spike > 1.8 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_20)
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (trend strength filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate +DM and -DM
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[0], tr])
    
    # Smooth the values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(plus_dm) < period:
        return np.zeros(n)
    
    smoothed_plus_dm = wilders_smoothing(plus_dm, period)
    smoothed_minus_dm = wilders_smoothing(minus_dm, period)
    smoothed_tr = wilders_smoothing(tr, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * smoothed_plus_dm / smoothed_tr
    minus_di = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    adx = wilders_smoothing(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams %R calculation (14-period)
    if len(close) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                          -50)  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (need ADX > 25 for trending market)
        trending = adx_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 from oversold with volume confirmation and trending market
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and volume_confirmation[i] and trending:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought with volume confirmation and trending market
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and volume_confirmation[i] and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R rises above -20 (overbought) OR market becomes ranging (ADX < 20)
            if williams_r[i] > -20 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -80 (oversold) OR market becomes ranging (ADX < 20)
            if williams_r[i] < -80 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals