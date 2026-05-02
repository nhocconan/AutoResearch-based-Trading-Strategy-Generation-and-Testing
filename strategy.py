#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme with 1d EMA34 trend filter and volume spike
# Targets 20-50 trades per year (80-200 total over 4 years) to minimize fee drag
# Williams %R identifies overbought/oversold conditions (below -80 for long, above -20 for short)
# 1d EMA34 ensures alignment with daily trend (avoid counter-trend trades)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in both bull and bear: trend filter prevents counter-trend trades, volume confirms validity

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for prior day calculation
        return np.zeros(n)
    
    # Calculate Williams %R from prior 1d bar (14-period lookback)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We use prior 14 days of data to avoid look-ahead
    high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().shift(1).values
    low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().shift(1).values
    close_1d = df_1d['close'].shift(1).values  # Prior day close
    
    # Avoid division by zero
    denominator = high_14 - low_14
    williams_r = np.where(denominator != 0, ((high_14 - close_1d) / denominator) * -100, -50)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().shift(1).values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1d EMA34
            if (williams_r_aligned[i] > -20 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 1d EMA34
            if (williams_r_aligned[i] < -80 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals