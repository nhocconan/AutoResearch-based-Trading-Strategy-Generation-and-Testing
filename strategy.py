#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1d Volume Spike + ADX Regime Filter
# Long when price breaks above Camarilla R3 with volume spike and ADX > 25
# Short when price breaks below Camarilla S3 with volume spike and ADX > 25
# Exit when ADX < 20 (range regime) or price returns to Camarilla Pivot point
# Uses discrete position sizing (0.25) to limit fee drag.
# Camarilla levels provide intraday support/resistance, volume confirms breakout strength.
# ADX ensures we only trade in trending markets to avoid false breakouts in ranges.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to avoid overtrading.

name = "12h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADX_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R4 = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    R2 = close_1d + ((high_1d - low_1d) * 1.1 / 6)
    R1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    P = (high_1d + low_1d + close_1d) / 3
    S1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    S2 = close_1d - ((high_1d - low_1d) * 1.1 / 6)
    S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    S4 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    R3_1d = align_htf_to_ltf(prices, df_1d, R3)
    S3_1d = align_htf_to_ltf(prices, df_1d, S3)
    P_1d = align_htf_to_ltf(prices, df_1d, P)
    
    # Get 1d volume for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate ADX (14-period) on 12h timeframe
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # ADX and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or np.isnan(P_1d[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol_ratio = vol_ratio_aligned[i]
        curr_adx = adx[i]
        curr_R3 = R3_1d[i]
        curr_S3 = S3_1d[i]
        curr_P = P_1d[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: ADX < 20 (range regime) OR price returns to pivot point
            if curr_adx < 20.0 or curr_close <= curr_P:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (range regime) OR price returns to pivot point
            if curr_adx < 20.0 or curr_close >= curr_P:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike condition: current volume > 1.5 * 20-day average
            volume_spike = curr_vol_ratio > 1.5
            
            # Long when price breaks above R3 with volume spike and ADX > 25 (trending)
            if curr_close > curr_R3 and volume_spike and curr_adx > 25.0:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S3 with volume spike and ADX > 25 (trending)
            elif curr_close < curr_S3 and volume_spike and curr_adx > 25.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals