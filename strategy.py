#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R with daily ADX trend filter and volume spike
# Long when Williams %R < -80 (oversold), daily ADX > 25 (trending), and volume spike
# Short when Williams %R > -20 (overbought), daily ADX > 25, and volume spike
# Williams %R identifies exhaustion points in trending markets
# ADX ensures we only trade in trending conditions (avoids chop)
# Volume spike confirms participation; avoids false signals
# Targets 50-150 total trades over 4 years (12-37/year) with disciplined entries

name = "6h_WilliamsR_ADX_Trend_Volume"
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
    
    # Get daily data once for ADX and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14) for trend filter
    df_1h = df_1d  # daily data
    high_1d = df_1h['high'].values
    low_1d = df_1h['low'].values
    close_1d = df_1h['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    tr_period = 14
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial values
    atr[tr_period-1] = np.mean(tr[:tr_period])
    plus_dm_smooth[tr_period-1] = np.mean(plus_dm[:tr_period])
    minus_dm_smooth[tr_period-1] = np.mean(minus_dm[:tr_period])
    
    # Wilder's smoothing
    for i in range(tr_period, len(tr)):
        atr[i] = (atr[i-1] * (tr_period-1) + tr[i]) / tr_period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (tr_period-1) + plus_dm[i]) / tr_period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (tr_period-1) + minus_dm[i]) / tr_period
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx) | (plus_di + minus_di == 0)] = 0
    
    # ADX
    adx = np.zeros_like(dx)
    adx[2*tr_period-1] = np.mean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period, len(dx)):
        adx[i] = (adx[i-1] * (tr_period-1) + dx[i]) / tr_period
    
    adx = adx[tr_period-1:]  # align with original arrays
    
    # Calculate Williams %R(14) on daily
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # Align ADX and Williams %R to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx[13:])  # skip first 13 for ADX warmup
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r[13:])  # skip first 13 for WR warmup
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        williams_r_val = williams_r_aligned[i]
        price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: oversold, strong trend, volume spike
            if williams_r_val < -80 and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought, strong trend, volume spike
            elif williams_r_val > -20 and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: overbought or trend weakens
            if williams_r_val > -20 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: oversold or trend weakens
            if williams_r_val < -80 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals