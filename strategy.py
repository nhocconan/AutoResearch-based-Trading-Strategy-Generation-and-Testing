#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Alligator (Jaw/Teeth/Lips) to identify trend absence (all lines intertwined = chop) vs presence (lines separated = trend).
# 1d EMA50 for higher timeframe trend alignment.
# Volume confirmation (>1.3x 20-bar avg) to reduce false signals.
# Session filter (08-20 UTC) for liquidity.
# Discrete position sizing at ±0.25 to manage fee drag on 12h timeframe.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid excessive fees.
# Works in bull markets via trend continuation and in bear markets via volatility expansion capture during ranging periods.

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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h: SMAs of median price
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA as approximation for SMMA (simple moving average)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Alligator and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Alligator condition: lines separated = trend present
        # Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        is_uptrend = curr_jaw > curr_teeth and curr_teeth > curr_lips
        is_downtrend = curr_jaw < curr_teeth and curr_teeth < curr_lips
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend + price above Jaw + close > 1d EMA50 + volume spike
            if is_uptrend and curr_close > curr_jaw and curr_close > curr_ema_50_1d and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + price below Jaw + close < 1d EMA50 + volume spike
            elif is_downtrend and curr_close < curr_jaw and curr_close < curr_ema_50_1d and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Alligator lines intertwine (chop) or price crosses below Teeth
            jaws_teeth_cross = abs(curr_jaw - curr_teeth) < 0.001 * curr_close  # approximate intertwine
            teeth_lips_cross = abs(curr_teeth - curr_lips) < 0.001 * curr_close
            if jaws_teeth_cross or teeth_lips_cross or curr_close < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Alligator lines intertwine (chop) or price crosses above Teeth
            jaws_teeth_cross = abs(curr_jaw - curr_teeth) < 0.001 * curr_close
            teeth_lips_cross = abs(curr_teeth - curr_lips) < 0.001 * curr_close
            if jaws_teeth_cross or teeth_lips_cross or curr_close > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals