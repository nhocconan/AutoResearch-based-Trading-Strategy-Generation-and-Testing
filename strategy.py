#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extreme reversal with 1d volume spike confirmation and 1w ADX trend filter.
# Long when: 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-bar average AND 1w ADX > 25 (trending market)
# Short when: 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-bar average AND 1w ADX > 25 (trending market)
# Exit via opposing Williams %R signal: long exit when Williams %R > -50, short exit when Williams %R < -50
# Uses Williams %R for mean reversion in extremes, volume spike for confirmation, ADX to filter ranging markets
# Discrete sizing 0.25 balances return and fee drag. Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WilliamsR_1dVolumeSpike_1wADX_Trend_v1"
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    williams_r = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i < 13:
            williams_r[i] = np.nan
        else:
            highest_high = np.max(high_1d[i-13:i+1])
            lowest_low = np.min(low_1d[i-13:i+1])
            if highest_high == lowest_low:
                williams_r[i] = -50.0
            else:
                williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
    
    # Align 1d Williams %R to 6h timeframe (completed 1d bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1w ADX(14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR and DM
    tr_14 = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.where(tr_14 != 0, dm_plus_14 / tr_14 * 100, 0)
    di_minus = np.where(tr_14 != 0, dm_minus_14 / tr_14 * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Williams %R, ADX, volume calculations)
    start_idx = max(20, 14, 14) + 5  # Williams %R(14), ADX(14), volume MA(20) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R oversold (< -80) with volume spike AND trending market (ADX > 25)
            if williams_r_aligned[i] < -80 and volume_spike[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) with volume spike AND trending market (ADX > 25)
            elif williams_r_aligned[i] > -20 and volume_spike[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -50 (reversing from oversold)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (reversing from overbought)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals