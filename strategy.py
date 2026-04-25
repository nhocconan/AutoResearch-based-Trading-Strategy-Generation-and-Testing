#!/usr/bin/env python3
"""
1d Williams %R Reversal + 1w EMA34 Trend + Volume Spike
Hypothesis: Williams %R identifies overbought/oversold conditions on daily timeframe.
In strong trends (price > 1w EMA34), reversals from extreme %R levels offer high-probability entries.
Volume spike confirms institutional participation. Works in both bull and bear markets:
- Bull: Buy oversold reversals (%R < -80) in uptrend (price > 1w EMA34)
- Bear: Sell overbought reversals (%R > -20) in downtrend (price < 1w EMA34)
Designed for 1d timeframe with tight entry conditions to achieve 7-25 trades/year per symbol,
minimizing fee drag while capturing meaningful reversals in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and 1w data for EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - df_1d['close'].values) / (highest_high - lowest_low)) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 15m timeframe (no extra delay needed for %R)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate EMA34 on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 15m timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R (14), EMA (34), volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        williams_r_val = williams_r_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Williams %R oversold (< -80) AND volume spike AND price > EMA34 (uptrend)
            long_entry = (williams_r_val < -80) and vol_spike and (curr_close > ema_trend)
            # Short: Williams %R overbought (> -20) AND volume spike AND price < EMA34 (downtrend)
            short_entry = (williams_r_val > -20) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Williams %R returns above -50 (neutral) OR price crosses below EMA34 (trend change)
            if (williams_r_val > -50) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R returns below -50 (neutral) OR price crosses above EMA34 (trend change)
            if (williams_r_val < -50) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Reversal_EMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0