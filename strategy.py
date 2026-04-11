#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_fibonacci_retracement_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate daily pivot points (classic)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Shift by 1 to use only completed daily bars
    pivot = np.roll(pivot, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r2 = np.roll(r2, 1)
    s2 = np.roll(s2, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    pivot[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    r2[0] = np.nan
    s2[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 6-period ATR for dynamic sizing and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Price position relative to pivot and S1/R1
        near_support = price_low <= s1_aligned[i] + 0.5 * atr_val and price_close > s1_aligned[i]
        near_resistance = price_high >= r1_aligned[i] - 0.5 * atr_val and price_close < r1_aligned[i]
        
        # Breakout conditions with volume
        breakout_up = price_close > r1_aligned[i] and volume_confirmed
        breakout_down = price_close < s1_aligned[i] and volume_confirmed
        
        # Retracement entry conditions
        long_signal = volume_confirmed and near_support and price_close > pivot_aligned[i]
        short_signal = volume_confirmed and near_resistance and price_close < pivot_aligned[i]
        
        # Exit when price moves to opposite S1/R1 level
        exit_long = position == 1 and price_close >= r1_aligned[i]
        exit_short = position == -1 and price_close <= s1_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily pivot point retracement strategy on 6h timeframe.
# Uses classic pivot points (P, R1, S1, R2, S2, R3, S3) calculated from daily OHLC.
# Enters long near S1 support when price shows rejection (wick below S1 but close above it)
# with volume confirmation (>1.3x average). Enters short near R1 resistance when price
# shows rejection (wick above R1 but close below it) with volume confirmation.
# Exits when price reaches opposite R1/S1 level. Works in both ranging and trending
# markets by fading extremes and capturing mean reversion. Target: 50-150 total trades
# over 4 years (12-37/year) to minimize fee drag on 6h timeframe. Pivot points act as
# natural support/resistance levels where institutional order flow often concentrates.