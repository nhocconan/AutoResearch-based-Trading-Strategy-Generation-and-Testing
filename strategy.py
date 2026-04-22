#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d ADX trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions (> -20 = overbought, < -80 = oversold).
# Reversal from extremes with volume spike (>1.5x 20-period average) and trend alignment 
# (price > 1d ADX > 25 for longs, price < 1d ADX > 25 for shorts) captures mean reversion 
# in ranging markets while avoiding false signals in strong trends. Designed for low trade 
# frequency (~20-40/year) to minimize fee decay. Works in both bull and bear markets by 
# using ADX to filter trend strength and Williams %R for mean reversion signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on 1d data
    # True Range (TR)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initial values (simple average of first 14 periods)
    atr[13] = np.mean(tr[1:14])
    plus_dm_smooth[13] = np.mean(plus_dm[1:14])
    minus_dm_smooth[13] = np.mean(minus_dm[1:14])
    
    # Wilder's smoothing for remaining periods
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * 13 + plus_dm[i]) / 14
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.where(atr != 0, plus_dm_smooth / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_dm_smooth / atr * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = np.zeros_like(dx)
    adx[27] = np.mean(dx[14:28])  # First ADX is average of first 14 DX values
    for i in range(28, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Calculate Williams %R on 4h data (14-period)
    highest_high = pd.Series(prices['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(prices['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - prices['close'].values) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # Avoid division by zero
    
    # Align 1d ADX to 4h timeframe (waits for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        wr = williams_r[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        # ADX filter: trend strength > 25
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 from below + strong trend + volume confirmation
            if i > 0 and williams_r[i-1] <= -80 and wr > -80 and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 from above + strong trend + volume confirmation
            elif i > 0 and williams_r[i-1] >= -20 and wr < -20 and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R reaches overbought (-20) or trend weakens
                if wr >= -20 or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R reaches oversold (-80) or trend weakens
                if wr <= -80 or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_ADX25_Volume"
timeframe = "4h"
leverage = 1.0