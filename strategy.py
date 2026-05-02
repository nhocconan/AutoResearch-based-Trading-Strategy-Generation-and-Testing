#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme with 1w EMA34 trend filter and volume spike
# Targets 7-25 trades per year (30-100 total over 4 years) to minimize fee drag
# Williams %R identifies overbought/oversold conditions (above -20 = overbought, below -80 = oversold)
# 1w EMA34 ensures alignment with weekly trend (avoid counter-trend trades)
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in both bull and bear: trend filter prevents counter-trend trades, volume confirms validity
# Discrete position sizing 0.25 balances exposure and risk

name = "1d_WilliamsR_Extreme_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Williams %R calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 1 week for calculation
        return np.zeros(n)
    
    # Calculate Williams %R on 1w data (14-period lookback)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Above -20 = overbought, Below -80 = oversold
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    highest_high = high_1w.rolling(window=14, min_periods=14).max().values
    lowest_low = low_1w.rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1w) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align Williams %R to 1d timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Load 1w data ONCE before loop for EMA34 trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA34 AND volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA34 AND volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1w EMA34
            if (williams_r_aligned[i] > -20 or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 1w EMA34
            if (williams_r_aligned[i] < -80 or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals