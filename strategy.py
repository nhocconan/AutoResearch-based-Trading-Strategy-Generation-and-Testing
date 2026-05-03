#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above upper BB(20,2) with volume > 2x 20-bar avg and 1d ADX > 25 (trending)
# Short when price breaks below lower BB(20,2) with volume > 2x 20-bar avg and 1d ADX > 25 (trending)
# Exit when price returns to middle BB(20) or volume drops below average
# Bollinger Squeeze captures low volatility breakouts that often trend. ADX filter ensures we only trade in trending regimes (works in both bull and bear).
# Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete sizing (0.25) to minimize fee churn.

name = "6h_BollingerSqueeze_Breakout_1dADX_Volume_v1"
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
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, np.nan)
    di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), np.nan)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands (20,2) on 6h
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20, 20) + 1  # ADX(1d) + BB(20) + volume MA(20) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(adx_aligned[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper BB with volume spike and 1d ADX > 25 (trending)
            if (close[i] > bb_upper[i] and 
                volume_spike[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower BB with volume spike and 1d ADX > 25 (trending)
            elif (close[i] < bb_lower[i] and 
                  volume_spike[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to middle BB or volume drops below average
            if (close[i] <= bb_middle[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price returns to middle BB or volume drops below average
            if (close[i] >= bb_middle[i] or 
                volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals