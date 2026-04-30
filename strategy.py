#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation (>1.5x 30-bar avg).
# Uses 1d Williams Alligator for structure and trend direction, 1d EMA50 for higher timeframe confirmation.
# Volume confirmation filters weak breakouts. Session filter (08-20 UTC) avoids low-liquidity periods.
# Discrete position sizing at ±0.25 to balance return and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag on 12h timeframe.
# Williams Alligator identifies trending vs ranging markets: when lines are intertwined (no trend), we avoid entries.
# In strong trends (lines separated and aligned), we trade in direction of the alligator's mouth.
# Works in bull markets via trend continuation and in bear markets via mean-reversion when alligator lines converge.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_Session_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Williams Alligator and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Williams Alligator (SMMA = smoothed moving average)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = (high_1d + low_1d) / 2  # Williams Alligator uses median price
    
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.5 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for EMA50 and Alligator (max shift is 8, so 50+8=58, round to 60)
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw_1d_aligned[i]
        curr_teeth = teeth_1d_aligned[i]
        curr_lips = lips_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Alligator trend condition: lines are separated and aligned
        # Bullish trend: Lips > Teeth > Jaw (green above red above blue)
        # Bearish trend: Jaw > Teeth > Lips (blue above red above green)
        is_bullish_aligned = curr_lips > curr_teeth and curr_teeth > curr_jaw
        is_bearish_aligned = curr_jaw > curr_teeth and curr_teeth > curr_lips
        
        if position == 0:  # Flat - look for new entries
            # Long: price above EMA50, bullish alligator alignment, volume spike, in session
            if (curr_close > curr_ema_50_1d and 
                is_bullish_aligned and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50, bearish alligator alignment, volume spike, in session
            elif (curr_close < curr_ema_50_1d and 
                  is_bearish_aligned and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: alligator lines converge (trend weakening) or price breaks below teeth
            # Exit when lips cross below teeth (green crosses below red) - trend losing momentum
            if curr_lips < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: alligator lines converge (trend weakening) or price breaks above teeth
            # Exit when lips cross above teeth (green crosses above red) - trend losing momentum
            if curr_lips > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals