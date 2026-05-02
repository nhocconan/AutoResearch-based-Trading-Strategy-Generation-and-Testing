#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX regime + volume spike
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams %R(14) < -80 = oversold (long bias), > -20 = overbought (short bias)
# 1d ADX(14) > 25 = trending regime (allows entries), ADX < 20 = ranging (blocks entries)
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via buying oversold dips in uptrend and bear markets via selling overbought rallies in downtrend
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_WilliamsR_Extreme_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # True Range components
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'] - np.roll(df_1d['high'], 1)) > (np.roll(df_1d['low'], 1) - df_1d['low']),
                       np.maximum(df_1d['high'] - np.roll(df_1d['high'], 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'], 1) - df_1d['low']) > (df_1d['high'] - np.roll(df_1d['high'], 1)),
                        np.maximum(np.roll(df_1d['low'], 1) - df_1d['low'], 0), 0)
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initialize first values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder's smoothing
    for i in range(period, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
        dm_plus_smooth[i] = dm_plus_smooth[i-1] * (1 - alpha) + dm_plus[i] * alpha
        dm_minus_smooth[i] = dm_minus_smooth[i-1] * (1 - alpha) + dm_minus[i] * alpha
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.mean(dx[period-1:2*period])  # First ADX value
    
    for i in range(2*period, len(dx)):
        adx[i] = adx[i-1] * (1 - alpha) + dx[i] * alpha
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Calculate 6h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(30, 20, 14)  # ADX needs 30, volume MA needs 20, WilliamsR needs 14
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if (williams_r[i] < -80 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif (williams_r[i] > -20 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR ADX < 20 (ranging) OR volume drops
            if (williams_r[i] > -20 or 
                adx_aligned[i] < 20 or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR ADX < 20 (ranging) OR volume drops
            if (williams_r[i] < -80 or 
                adx_aligned[i] < 20 or 
                not volume_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals