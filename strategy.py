#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d/1w timeframes for regime and structure.
# Uses 1d Williams %R (14) for overbought/oversold conditions combined with 1w EMA50 trend filter.
# Enters on reversals from extreme %R levels when price breaks recent 6h high/low with volume confirmation.
# Designed to work in both bull and bear markets by requiring alignment with weekly trend.
# Target: 15-30 trades per year to minimize fee drag and improve generalization.
name = "6h_WilliamsR_1wEMA50_VolumeReversal"
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Align 1w EMA50 to 6h timeframe
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: spike above 2.0x 24-period average (1 day of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 6-period high/low for breakout confirmation
    high_6 = pd.Series(high).rolling(window=6, min_periods=6).max().values
    low_6 = pd.Series(low).rolling(window=6, min_periods=6).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24, 6)  # Wait for EMA50, volume MA, and 6-period high/low
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema_50_6h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_6[i]) or np.isnan(low_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: Williams %R below -80 (oversold), price breaks above 6-period high, weekly uptrend
            if (williams_r_6h[i] < -80 and 
                close[i] > high_6[i] and 
                close[i] > ema_50_6h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R above -20 (overbought), price breaks below 6-period low, weekly downtrend
            elif (williams_r_6h[i] > -20 and 
                  close[i] < low_6[i] and 
                  close[i] < ema_50_6h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R rises above -50 or trend reversal
            if williams_r_6h[i] > -50 or close[i] < ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R falls below -50 or trend reversal
            if williams_r_6h[i] < -50 or close[i] > ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals