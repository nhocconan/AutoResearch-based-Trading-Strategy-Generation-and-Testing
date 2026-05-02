#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band %B + 1d ADX regime filter + volume confirmation
# Bollinger %B measures price position within bands: %B = (price - lower) / (upper - lower)
# %B < 0 = oversold (long signal), %B > 1 = overbought (short signal)
# 1d ADX > 25 indicates trending market (fade extremes less), ADX < 20 indicates ranging (fade extremes more)
# Volume confirmation (1.5x 20-period average) ensures participation
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to regime via ADX

name = "6h_BBpercentB_1dADXRegime_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for ADX regime and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_len = 20
    bb_mult = 2.0
    basis = pd.Series(df_1d['close']).rolling(window=bb_len, min_periods=bb_len).mean().values
    dev = bb_mult * pd.Series(df_1d['close']).rolling(window=bb_len, min_periods=bb_len).std().values
    upper = basis + dev
    lower = basis - dev
    
    # Calculate 1d Bollinger %B: %B = (close - lower) / (upper - lower)
    # Avoid division by zero
    bb_range = upper - lower
    bb_range_safe = np.where(bb_range == 0, 1e-10, bb_range)
    percent_b = (df_1d['close'].values - lower) / bb_range_safe
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    # True Range
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values),
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)),
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    dm_plus[0] = dm_minus[0] = 0  # first bar
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    percent_b_aligned = align_htf_to_ltf(prices, df_1d, percent_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Bollinger Bands, ADX and volume MA)
    start_idx = 50  # max(20 for Bollinger, 34 for ADX) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(percent_b_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from 1d ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            if trending:
                # In trending market: only trade with stronger signals to avoid whipsaw
                # Long: %B < -0.2 (deep oversold) AND volume confirmation
                if (percent_b_aligned[i] < -0.2 and 
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: %B > 1.2 (deep overbought) AND volume confirmation
                elif (percent_b_aligned[i] > 1.2 and 
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:  # ranging or transition regime
                # In ranging market: fade extremes more aggressively
                # Long: %B < 0 (oversold) AND volume confirmation
                if (percent_b_aligned[i] < 0 and 
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: %B > 1 (overbought) AND volume confirmation
                elif (percent_b_aligned[i] > 1 and 
                      volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending long when %B rises above 0.5 (recovering from oversold)
                if percent_b_aligned[i] > 0.5:
                    exit_signal = True
            else:
                # Exit ranging long when %B rises above 0.8 (approaching fair value)
                if percent_b_aligned[i] > 0.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if trending:
                # Exit trending short when %B falls below 0.5 (declining from overbought)
                if percent_b_aligned[i] < 0.5:
                    exit_signal = True
            else:
                # Exit ranging short when %B falls below 0.2 (approaching fair value)
                if percent_b_aligned[i] < 0.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals