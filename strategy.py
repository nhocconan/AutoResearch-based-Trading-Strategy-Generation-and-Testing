#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d/1w Donchian breakouts with ATR filter and volume confirmation
# 1d/1w Donchian channels (20-period) identify major support/resistance
# Breakouts above upper channel or below lower channel with volume confirmation (1.5x 20-period average)
# ATR filter ensures sufficient volatility (ATR > 20-period ATR mean)
# Fixed position size of 0.25 to balance return and drawdown
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_1w_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    highest_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w Donchian channels (20-period)
    highest_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w ATR (14-period) for volatility filtering
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr1_1w[0]  # First period has no previous close
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    highest_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_1d)
    lowest_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    highest_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_1w)
    lowest_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_1d_aligned[i]) or np.isnan(lowest_1d_aligned[i]) or
            np.isnan(highest_1w_aligned[i]) or np.isnan(lowest_1w_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            atr_1d_aligned[i] <= 0 or atr_1w_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility filter: only trade when both 1d and 1w ATR are above their 20-period averages
        atr_ma_20_1d = pd.Series(atr_1d_aligned).rolling(window=20, min_periods=20).mean()
        atr_ma_20_1w = pd.Series(atr_1w_aligned).rolling(window=20, min_periods=20).mean()
        if len(atr_ma_20_1d) > i and len(atr_ma_20_1w) > i:
            vol_filter = (atr_1d_aligned[i] > atr_ma_20_1d.iloc[i]) and (atr_1w_aligned[i] > atr_ma_20_1w.iloc[i])
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to midpoint of 1d or 1w channel
            midpoint_1d = (highest_1d_aligned[i] + lowest_1d_aligned[i]) / 2.0
            midpoint_1w = (highest_1w_aligned[i] + lowest_1w_aligned[i]) / 2.0
            if close[i] < midpoint_1d or close[i] < midpoint_1w:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midpoint of 1d or 1w channel
            midpoint_1d = (highest_1d_aligned[i] + lowest_1d_aligned[i]) / 2.0
            midpoint_1w = (highest_1w_aligned[i] + lowest_1w_aligned[i]) / 2.0
            if close[i] > midpoint_1d or close[i] > midpoint_1w:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above upper channel (buy)
                if close[i] > highest_1d_aligned[i] or close[i] > highest_1w_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below lower channel (sell)
                elif close[i] < lowest_1d_aligned[i] or close[i] < lowest_1w_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals