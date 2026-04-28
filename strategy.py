#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Trend_1dEMA34_VolumeSpike_HT
Hypothesis: Target high-probability breakouts at daily Camarilla R3/S3 levels with 1-day EMA34 trend filter and volume spike confirmation on 4h timeframe.
Designed to work in both bull and bear markets by using trend alignment and volatility filters to reduce false signals. Targets 20-50 trades/year through strict confluence requirements.
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
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels (more extreme than R1/S1)
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
    
    # Volume confirmation: current volume > 2.5x 20-period average (more stringent)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with trend alignment and volume spike
        # Long: price breaks above R3 + 1d uptrend + volume spike
        long_entry = (close[i] > R3_aligned[i] and 
                     d1_uptrend[i] and 
                     volume_spike[i])
        
        # Short: price breaks below S3 + 1d downtrend + volume spike
        short_entry = (close[i] < S3_aligned[i] and 
                      d1_downtrend[i] and 
                      volume_spike[i])
        
        # Exit on opposite level break with volume spike
        long_exit = close[i] < S3_aligned[i] and volume_spike[i]
        short_exit = close[i] > R3_aligned[i] and volume_spike[i]
        
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

name = "4h_Camarilla_R3_S3_Trend_1dEMA34_VolumeSpike_HT"
timeframe = "4h"
leverage = 1.0