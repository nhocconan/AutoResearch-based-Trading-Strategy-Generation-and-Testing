#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with Volume Confirmation and Daily ADX Filter
# Strategy looks for low volatility periods (Bollinger Band squeeze) followed by breakouts,
# with volume confirmation and trend strength (ADX > 25) to filter false breakouts.
# Works in both bull and bear markets by trading breakouts in the direction of the trend.
# Target: 50-150 total trades over 4 years (12-38/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) - squeeze detection
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * std
    lower_band = sma - bb_std * std
    bb_width = (upper_band - lower_band) / sma  # Normalized width
    
    # Bollinger Band squeeze: width below 20-period average width
    avg_bb_width = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < 0.8 * avg_bb_width  # Squeeze when width is 20% below average
    
    # Breakout signals
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze[i]) or np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: Bollinger Band squeeze breakout up + volume confirmation + ADX > 25
        if (squeeze[i] and breakout_up[i] and volume_confirm[i] and 
            adx_aligned[i] > 25 and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bollinger Band squeeze breakout down + volume confirmation + ADX > 25
        elif (squeeze[i] and breakout_down[i] and volume_confirm[i] and 
              adx_aligned[i] > 25 and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or ADX < 20 (ranging market) or volatility expansion
        elif position == 1 and (breakout_down[i] or adx_aligned[i] < 20 or 
                                bb_width[i] > 2 * avg_bb_width[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up[i] or adx_aligned[i] < 20 or 
                                 bb_width[i] > 2 * avg_bb_width[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Squeeze_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0