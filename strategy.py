#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band width squeeze breakout with 1d ADX trend filter and volume confirmation.
# Bollinger Band width (BBW) identifies low volatility periods (squeeze).
# Breakout occurs when price moves outside Bollinger Bands after a squeeze.
# 1d ADX > 25 filters for trending markets to avoid false breakouts in ranging conditions.
# Volume > 1.5x 20-period average confirms breakout strength.
# Works in both bull and bear markets by trading breakouts in the direction of the 1d trend.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_BBW_Squeeze_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width squeeze: BBW < 50th percentile of past 50 bars
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.5).values
    bb_squeeze = bb_width < bb_width_percentile
    
    # Calculate 1d ADX for trend filter
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = np.where((high_1d - high_1d.shift(1)) > (low_1d.shift(1) - low_1d), 
                       np.maximum(high_1d - high_1d.shift(1), 0), 0)
    dm_minus = np.where((low_1d.shift(1) - low_1d) > (high_1d - high_1d.shift(1)), 
                        np.maximum(low_1d.shift(1) - low_1d, 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_mid[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Look for breakout after squeeze
            if bb_squeeze[i-1] and not bb_squeeze[i]:  # Squeeze just ended
                # Long: price breaks above upper band AND trending up AND volume confirmed
                if close[i] > bb_upper[i] and trending and volume_confirmed[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower band AND trending down AND volume confirmed
                elif close[i] < bb_lower[i] and trending and volume_confirmed[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band OR squeeze returns
            if close[i] < bb_mid[i] or bb_squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band OR squeeze returns
            if close[i] > bb_mid[i] or bb_squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals