#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and ADX>25 trend filter.
# Camarilla pivots provide institutional support/resistance levels. Breakout above R3
# or below S3 with volume confirmation indicates strong institutional interest.
# Filtered by 1d ADX > 25 to ensure we only trade in trending markets, avoiding
# false breakouts in ranging conditions. Works in both bull and bear markets by
# capturing breakouts in the direction of the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "12h_Camarilla_R3S3_Breakout_1dADX25_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    df_1d_volume = df_1d['volume'].values
    
    # True Range
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low), 
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)), 
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (atr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # 1d volume spike: current volume > 1.5x 20-period MA
    vol_ma_20_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d_volume > (1.5 * vol_ma_20_1d)
    
    # Align 1d indicators to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d_high + df_1d_low + df_1d_close) / 3.0
    range_1d = df_1d_high - df_1d_low
    
    # Camarilla levels
    R3 = typical_price + (range_1d * 1.1 / 4)
    S3 = typical_price - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # Convert back to boolean
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        
        # Determine trend regime
        is_trending = adx_val > 25
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above R3 AND trending AND volume spike
            if close_val > R3_val and is_trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 AND trending AND volume spike
            elif close_val < S3_val and is_trending and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close moves back below R3 OR trend weakens (ADX < 20)
            if close_val < R3_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close moves back above S3 OR trend weakens (ADX < 20)
            if close_val > S3_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals