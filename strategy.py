#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA(34) trend filter and volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# In trending markets, the Alligator lines are aligned and separated (Green > Red > Blue for uptrend)
# In ranging markets, the Alligator lines are intertwined
# 1w EMA(34) ensures alignment with weekly trend to avoid counter-trend trades
# Volume spike (>2.0x 20-period EMA) filters low-probability signals
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and improve generalization

name = "1d_WilliamsAlligator_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator: Three smoothed moving averages
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    close_series = pd.Series(close)
    
    # Lips (5-period SMMA, shift 3)
    lips = close_series.ewm(alpha=1/5, adjust=False).mean()
    lips = lips.shift(3)
    
    # Teeth (8-period SMMA, shift 5)
    teeth = close_series.ewm(alpha=1/8, adjust=False).mean()
    teeth = teeth.shift(5)
    
    # Jaw (13-period SMMA, shift 8)
    jaw = close_series.ewm(alpha=1/13, adjust=False).mean()
    jaw = jaw.shift(8)
    
    lips_vals = lips.values
    teeth_vals = teeth.values
    jaw_vals = jaw.values
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(lips_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(jaw_vals[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1w trend filter
        # Long: Lips > Teeth > Jaw (Alligator aligned up) + price above 1w EMA34 + volume spike
        # Short: Lips < Teeth < Jaw (Alligator aligned down) + price below 1w EMA34 + volume spike
        if position == 0:
            if (lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i] and 
                close[i] > ema_34_1w_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (lips_vals[i] < teeth_vals[i] and teeth_vals[i] < jaw_vals[i] and 
                  close[i] < ema_34_1w_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines intertwine (Lips < Teeth or Teeth < Jaw) OR price below 1w EMA34
            if (lips_vals[i] < teeth_vals[i] or teeth_vals[i] < jaw_vals[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines intertwine (Lips > Teeth or Teeth > Jaw) OR price above 1w EMA34
            if (lips_vals[i] > teeth_vals[i] or teeth_vals[i] > jaw_vals[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals