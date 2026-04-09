#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d Donchian breakouts with ATR filtering and session filter
# 4h/1d Donchian channels (20-period) identify major support/resistance
# Breakouts above upper channel or below lower channel with volume confirmation
# ATR filter ensures sufficient volatility for meaningful moves
# Session filter (08-20 UTC) reduces noise trades
# Fixed position size of 0.20 to control drawdown
# Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)

name = "1h_4h_1d_donchian_atr_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Donchian channels (20-period)
    highest_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR (14-period) for volatility filtering
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filtering
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 1h timeframe
    highest_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_4h)
    lowest_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    highest_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_1d)
    lowest_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(highest_4h_aligned[i]) or np.isnan(lowest_4h_aligned[i]) or
            np.isnan(highest_1d_aligned[i]) or np.isnan(lowest_1d_aligned[i]) or
            np.isnan(atr_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_4h_aligned[i] <= 0 or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.3x average 1h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        # Volatility filter: only trade when both 4h and 1d ATR are above their 30-period averages
        atr_ma_30_4h = pd.Series(atr_4h_aligned).rolling(window=30, min_periods=30).mean()
        atr_ma_30_1d = pd.Series(atr_1d_aligned).rolling(window=30, min_periods=30).mean()
        if len(atr_ma_30_4h) > i and len(atr_ma_30_1d) > i:
            vol_filter = (atr_4h_aligned[i] > atr_ma_30_4h.iloc[i]) and (atr_1d_aligned[i] > atr_ma_30_1d.iloc[i])
        else:
            vol_filter = True  # Not enough data for MA, allow trading
            
        if not vol_filter:
            signals[i] = 0.0
            continue
        
        # Fixed position size to minimize fee churn
        position_size = 0.20
        
        if position == 1:  # Long position
            # Exit on retracement to midpoint of 4h or 1d channel
            midpoint_4h = (highest_4h_aligned[i] + lowest_4h_aligned[i]) / 2.0
            midpoint_1d = (highest_1d_aligned[i] + lowest_1d_aligned[i]) / 2.0
            if close[i] < midpoint_4h or close[i] < midpoint_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midpoint of 4h or 1d channel
            midpoint_4h = (highest_4h_aligned[i] + lowest_4h_aligned[i]) / 2.0
            midpoint_1d = (highest_1d_aligned[i] + lowest_1d_aligned[i]) / 2.0
            if close[i] > midpoint_4h or close[i] > midpoint_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume and volatility confirmation
            if volume_confirmed:
                # Breakout above upper channel (buy)
                if close[i] > highest_4h_aligned[i] or close[i] > highest_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout below lower channel (sell)
                elif close[i] < lowest_4h_aligned[i] or close[i] < lowest_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals