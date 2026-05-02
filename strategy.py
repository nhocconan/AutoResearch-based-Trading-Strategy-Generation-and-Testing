#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend identification and entry timing
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms breakout strength
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 15-40 trades/year (60-160 total over 4 years) to stay within fee drag limits
# Williams Alligator provides clear trend signals that work in both bull and bear markets

name = "4h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 4h (median prices)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # Green line
    
    # Calculate 4h volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator calculation)
    start_idx = 13
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator signals: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
            # Long: Lips > Teeth > Jaw (bullish alignment) AND above 1d EMA34 AND volume confirm
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND below 1d EMA34 AND volume confirm
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish (Lips < Teeth OR Teeth < Jaw) OR below 1d EMA34
            if (lips[i] < teeth[i] or teeth[i] < jaw[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish (Lips > Teeth OR Teeth > Jaw) OR above 1d EMA34
            if (lips[i] > teeth[i] or teeth[i] > jaw[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals