#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify
# trend direction and strength. Long when Lips > Teeth > Jaw (bullish alignment),
# Short when Lips < Teeth < Jaw (bearish alignment). Filtered by 1d ADX > 25 for
# trending markets only and volume > 1.5x 20-period MA to avoid false signals.
# Works in bull via long signals and bear via short signals when aligned with trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25.

name = "6h_WilliamsAlligator_1dADX25_Volume"
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
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
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 6h: three SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA is Wilder's smoothing: EMA with alpha=1/period
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().values
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().values
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values after roll are invalid, set to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Entry logic
        if position == 0:
            # Long: Bullish Alligator alignment AND trending AND volume spike
            if bullish_alignment and is_trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND trending AND volume spike
            elif bearish_alignment and is_trending and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR trend weakens (ADX < 20) OR volume drops
            if bearish_alignment or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR trend weakens (ADX < 20) OR volume drops
            if bullish_alignment or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals