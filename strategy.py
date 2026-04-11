#!/usr/bin/env python3
# 4h_1d_williams_alligator_ema_volume_v1
# Strategy: 4h Williams Alligator with 1d EMA trend and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Williams Alligator (Jaw, Teeth, Lips) identifies trends when lines are aligned and separated.
# In bull markets: Lips > Teeth > Jaw (uptrend). In bear markets: Lips < Teeth < Jaw (downtrend).
# Combined with 1d EMA50 trend filter and volume confirmation to avoid false signals.
# Designed for low trade frequency (<50/year) to minimize fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_williams_alligator_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 4h (13,8,5 periods with shifts 8,5,3)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Smoothed median price (typical price)
    typical_price = (high + low + close) / 3
    
    # Jaw (blue line): 13-period SMMA of typical price, shifted 8 bars
    jaw_raw = pd.Series(typical_price).rolling(window=jaw_period, min_periods=jaw_period).mean()
    jaw = jaw_raw.shift(8)
    
    # Teeth (red line): 8-period SMMA of typical price, shifted 5 bars
    teeth_raw = pd.Series(typical_price).rolling(window=teeth_period, min_periods=teeth_period).mean()
    teeth = teeth_raw.shift(5)
    
    # Lips (green line): 5-period SMMA of typical price, shifted 3 bars
    lips_raw = pd.Series(typical_price).rolling(window=lips_period, min_periods=lips_period).mean()
    lips = lips_raw.shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment: check if lines are properly ordered and separated
        # Bullish alignment: Lips > Teeth > Jaw (alligator waking up to eat)
        bullish_alignment = (lips_values[i] > teeth_values[i]) and (teeth_values[i] > jaw_values[i])
        # Bearish alignment: Lips < Teeth < Jaw (alligator sleeping)
        bearish_alignment = (lips_values[i] < teeth_values[i]) and (teeth_values[i] < jaw_values[i])
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: current 1d volume > 20-period average
        vol_confirm = vol_1d_aligned[i] > vol_avg_20_1d_aligned[i]
        
        # Entry conditions
        # Long: Bullish alligator alignment AND uptrend AND volume confirmation
        if bullish_alignment and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bearish alligator alignment AND downtrend AND volume confirmation
        elif bearish_alignment and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Alligator lines intertwine (market sleeping) or reverse alignment
        elif position == 1 and not bullish_alignment:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not bearish_alignment:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals