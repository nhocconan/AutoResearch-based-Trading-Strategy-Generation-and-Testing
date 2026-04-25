#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_VolumeSpike
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1-day ATR volatility filter and volume spike confirmation. 
In ranging markets: buy when price breaks above Camarilla R1 and daily ATR is elevated (vol expansion). 
In trending markets: sell when price breaks below Camarilla S3 and daily ATR is elevated. 
Requires volume > 2.0x 20-period average for confirmation to avoid false breakouts. 
Exit on opposite Camarilla level touch (R1 for shorts, S3 for longs) or when ATR contracts below threshold. 
Position size: 0.25 to limit drawdown. 
Target: 75-200 total trades over 4 years = 19-50/year. 
Uses 1-day ATR to filter for volatility expansion regimes, which improves breakout reliability in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, ATR, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ATR calculation
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Camarilla levels for each 1d bar
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    r3_1d = close_1d + (1.1 * hl_range_1d / 4)   # R3 = close + 1.1*(high-low)/4
    s3_1d = close_1d - (1.1 * hl_range_1d / 4)   # S3 = close - 1.1*(high-low)/4
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR(14) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: ATR > 0.5 * 20-period ATR average (volatility expansion)
        atr_ma_20 = pd.Series(atr_14_1d_aligned[:i+1]).rolling(window=20, min_periods=1).mean().iloc[-1] if i >= 20 else atr_14_1d_aligned[i]
        vol_filter = atr_14_1d_aligned[i] > 0.5 * atr_ma_20
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + volatility expansion + volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and vol_filter and volume_confirm
            
            # Short setup: price breaks below Camarilla S3 + volatility expansion + volume confirmation
            short_setup = (close[i] < s3_aligned[i]) and vol_filter and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Camarilla S1 (stop) OR volatility contracts
            if (close[i] <= s1_aligned[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR volatility contracts
            if (close[i] >= r1_aligned[i]) or (not vol_filter):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_VolumeSpike"
timeframe = "4h"
leverage = 1.0