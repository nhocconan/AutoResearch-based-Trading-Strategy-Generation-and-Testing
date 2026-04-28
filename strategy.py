#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Dyn
Hypothesis: Focus on breakouts at daily Camarilla R3/S3 levels with 1d trend filter and volume confirmation on 4h timeframe.
This version uses a dynamic volume threshold (3x ATR-based volume filter) and targets 20-50 trades/year by requiring multiple confluence factors (breakout, trend, volume).
Designed to work in both bull and bear markets by filtering for trend alignment and using volume spikes to confirm breakout strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all higher timeframe data to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend filter: price > EMA34 = bullish, < EMA34 = bearish
    d1_uptrend = close > ema_34_1d_aligned
    d1_downtrend = close < ema_34_1d_aligned
    
    # Volume confirmation: current volume > 3.0x 14-period ATR-based volume filter
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    atr_14_safe = np.where(atr_14 == 0, 1e-10, atr_14)
    # Volume normalized by ATR
    vol_atr_ratio = volume / atr_14_safe
    vol_atr_ma = pd.Series(vol_atr_ratio).rolling(window=14, min_periods=14).mean().values
    volume_surge = vol_atr_ratio > (vol_atr_ma * 3.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above R3 + 1d uptrend + volume surge
        long_entry = (close[i] > R3_aligned[i] and 
                     d1_uptrend[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S3 + 1d downtrend + volume surge
        short_entry = (close[i] < S3_aligned[i] and 
                      d1_downtrend[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Dyn"
timeframe = "4h"
leverage = 1.0