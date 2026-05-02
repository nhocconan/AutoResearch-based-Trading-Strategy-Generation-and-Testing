#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter
# Camarilla levels provide intraday support/resistance; breakouts above R3 or below S3 with volume confirm strong moves
# 1d ADX > 25 filters for trending markets to avoid false breakouts in ranging conditions
# Discrete position sizing 0.25 minimizes fee churn while allowing meaningful exposure
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by only taking breakouts in trending regimes (ADX > 25)

name = "4h_Camarilla_R3S3_Breakout_1dADX_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filter
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Camarilla levels from previous day's OHLC
    # Need to get previous 1d bar's OHLC for each 4h bar
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_open_1d = np.roll(df_1d['open'].values, 1)
    prev_close_1d[0] = prev_high_1d[0] = prev_low_1d[0] = prev_open_1d[0] = 0
    
    # Calculate Camarilla levels for previous 1d bar
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for ADX and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] <= 25:
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 with volume confirmation
            if close[i] > r3_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation
            elif close[i] < s3_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 (reversal signal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 (reversal signal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals