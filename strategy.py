# 48  #!/usr/bin/env python3
# Hypothesis: 4-hour Williams %R with 1-day ADX trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (from oversold) with ADX > 25 and volume > 1.5x average.
# Short when Williams %R crosses below -80 (from overbought) with ADX > 25 and volume > 1.5x average.
# Exit when Williams %R returns to the opposite extreme or volume drops below average.
# Williams %R captures momentum reversals, ADX filters for trending markets to avoid chop,
# and volume confirms strength. Target: 20-50 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ADX and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
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
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    dm_plus_smooth[13] = np.mean(dm_plus[1:14])
    dm_minus_smooth[13] = np.mean(dm_minus[1:14])
    
    # Wilder's smoothing
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI and DX
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    # ADX
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX after 2*period
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Williams %R (14-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close_4h) / (highest_high - lowest_low), -50)
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (4h close and 1d volume aligned)
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -20 (from oversold), ADX > 25, volume surge
            if (williams_r[i] > -20 and williams_r[i-1] <= -20 and
                adx_aligned[i] > 25 and
                vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -80 (from overbought), ADX > 25, volume surge
            elif (williams_r[i] < -80 and williams_r[i-1] >= -80 and
                  adx_aligned[i] > 25 and
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to opposite extreme or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R < -80 or volume < average
                if (williams_r[i] < -80 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R > -20 or volume < average
                if (williams_r[i] > -20 or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_ADX25_Volume1.5x"
timeframe = "4h"
leverage = 1.0