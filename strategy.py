#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian breakouts with ATR filtering and volume confirmation
# 1w Donchian channels (20-period) identify major weekly support/resistance
# Breakouts above upper channel or below lower channel with volume confirmation
# ATR filter ensures sufficient volatility for meaningful moves
# Fixed position size of 0.25 to control drawdown
# Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years)
# Works in both bull and bear markets by capturing strong directional moves

name = "1d_1w_donchian_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    highest_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w ATR (14-period) for volatility filtering
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 1d timeframe
    highest_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_1w)
    lowest_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Pre-compute volume confirmation (20-period average for 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_1w_aligned[i]) or np.isnan(lowest_1w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or
            atr_1w_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when 1w ATR is above its 20-period average
        if i >= 20:
            atr_ma_20 = pd.Series(atr_1w_aligned).rolling(window=20, min_periods=20).mean().iloc[i]
            vol_filter = atr_1w_aligned[i] > atr_ma_20
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to midpoint of 1w channel
            midpoint_1w = (highest_1w_aligned[i] + lowest_1w_aligned[i]) / 2.0
            if close[i] < midpoint_1w:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midpoint of 1w channel
            midpoint_1w = (highest_1w_aligned[i] + lowest_1w_aligned[i]) / 2.0
            if close[i] > midpoint_1w:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above upper channel (buy)
                if close[i] > highest_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below lower channel (sell)
                elif close[i] < lowest_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals