#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d ADX trend filter + 12h price action breakout.
# Uses 1d ADX > 25 to identify trending markets, then enters long on 12h close above
# prior 12h high + 0.5*ATR(1d) and short below prior 12h low - 0.5*ATR(1d).
# Volume confirmation ensures breakout strength. Works in bull/bear by capturing
# directional moves. Target: 50-150 total trades over 4 years (12-37/year).
name = "12h_1d_ADXTrend_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and ATR calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d timeframe
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR on 1d timeframe
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX and ATR to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for ADX/ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Only trade in trending markets (ADX > 25)
        if adx_1d_aligned[i] <= 25:
            # Exit position if not trending
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        # Calculate breakout levels using prior 12h bar + 0.5*ATR
        prev_high = high[i-1]
        prev_low = low[i-1]
        atr_val = atr_1d_aligned[i] * 0.5  # Half ATR for sensitivity
        
        long_breakout = prev_high + atr_val
        short_breakout = prev_low - atr_val
        
        if position == 0:
            # Long when price breaks above prior high + 0.5*ATR with volume
            if close[i] > long_breakout and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below prior low - 0.5*ATR with volume
            elif close[i] < short_breakout and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # Long position: exit when price falls below prior low - 0.5*ATR
            if close[i] < short_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above prior high + 0.5*ATR
            if close[i] > long_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals