#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h ADX25 regime filter and volume confirmation
# Long when: price breaks above upper BB(20,2) AND ADX > 25 (trending) AND volume > 1.8x 20-bar avg
# Short when: price breaks below lower BB(20,2) AND ADX > 25 AND volume > 1.8x 20-bar avg
# Exit when: price returns to middle BB OR ADX < 20 (range) OR volume drops below 1.2x avg
# Bollinger squeeze identifies low volatility precedng breakouts; ADX filters for trending markets only
# Works in both bull and bear markets by trading breakouts in the direction of the trend (via ADX)
# Target: 12-30 trades/year via strict entry conditions reducing false breakouts in ranging markets

name = "6h_BollingerSqueeze_Breakout_12hADX25_Regime_VolumeFilter_v1"
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
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    # Calculate ADX(14) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    # Smoothed DM
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where(np.isnan(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Prepend zeros for alignment (since we lost first bar in calculations)
    adx = np.concatenate([np.full(27, np.nan), adx])  # 14 (TR) + 14 (ADX smoothing) - 1
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate Bollinger Bands on 6h close
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width for squeeze detection (optional filter)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < 0.8 * bb_width_ma  # Squeeze when BB width is 20% below its 20-bar MA
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(volume_ma_20[i]) or np.isnan(bb_squeeze[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        adx_val = adx_aligned[i]
        squeeze = bb_squeeze[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when: price breaks above upper BB AND ADX > 25 (trending) AND volume confirmation
            # Optional: require Bollinger squeeze for higher quality breakouts
            if price > bb_upper[i] and adx_val > 25 and vol_conf:  # and squeeze:
                signals[i] = 0.25
                position = 1
            # Short when: price breaks below lower BB AND ADX > 25 AND volume confirmation
            elif price < bb_lower[i] and adx_val > 25 and vol_conf:  # and squeeze:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to middle BB OR ADX < 20 OR volume drops
            if price < bb_middle[i] or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price returns to middle BB OR ADX < 20 OR volume drops
            if price > bb_middle[i] or adx_val < 20 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals