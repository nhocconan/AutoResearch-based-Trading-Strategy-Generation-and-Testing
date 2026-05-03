#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d trend filter and volume confirmation.
# Elder Ray measures bull/bear strength relative to EMA13. Long when Bull Power > 0 and rising,
# Short when Bear Power < 0 and falling. Filtered by 1d ADX > 25 for trending markets only and
# volume > 1.3x 20-period MA to avoid false signals. Works in bull via long signals and bear via
# short signals when aligned with trend. Target: 50-150 total trades over 4 years (12-37/year).
# Discrete sizing: 0.25.

name = "6h_ElderRay_1dADX25_Volume"
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
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Slope of Bull/Bear Power (3-period EMA of the power)
    bull_power_slope = pd.Series(bull_power).ewm(span=3, adjust=False).mean().values
    bear_power_slope = pd.Series(bear_power).ewm(span=3, adjust=False).mean().values
    
    # Volume regime: current 6h volume > 1.3x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or np.isnan(vol_ma_20[i])):
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
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        bull_power_rising = bull_power_slope[i] > 0
        bear_power_falling = bear_power_slope[i] < 0
        
        # Entry logic
        if position == 0:
            # Long: Bull Power > 0 AND rising AND trending AND volume spike
            if bull_power_positive and bull_power_rising and is_trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND falling AND trending AND volume spike
            elif bear_power_negative and bear_power_falling and is_trending and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power < 0 AND falling OR trend weakens (ADX < 20) OR volume drops
            if bear_power_negative and bear_power_falling or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power > 0 AND rising OR trend weakens (ADX < 20) OR volume drops
            if bull_power_positive and bull_power_rising or is_ranging or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals