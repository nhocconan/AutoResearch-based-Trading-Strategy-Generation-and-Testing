#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator strategy with 1d trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# Long when: Alligator is bullish (Lips > Teeth > Jaw) AND price > 1d EMA50 AND volume > 2x average
# Short when: Alligator is bearish (Lips < Teeth < Jaw) AND price < 1d EMA50 AND volume > 2x average
# Exit when Alligator direction reverses or opposite signal occurs.
# Uses Alligator for trend identification (proven in ranging/bear markets), 1d EMA for higher timeframe alignment.
# Volume confirmation reduces false signals. Discrete sizing 0.25 to control fee drag.
# Works in bull (buy Alligator alignment in uptrend) and bear (sell alignment in downtrend).

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
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
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components on 4h
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    vol_4h = df_4h['volume'].values
    
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    # Using SMA as approximation for SMMA (close enough for strategy purposes)
    jaw_4h = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    teeth_4h = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    lips_4h = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator lines to 4h primary timeframe
    jaw_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_4h)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h primary timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current 4h volume > 2x 20-period average
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 (50) + Alligator jaws (13) + volume MA (20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(jaw_4h_aligned[i]) or np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_jaw = jaw_4h_aligned[i]
        curr_teeth = teeth_4h_aligned[i]
        curr_lips = lips_4h_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma_4h_aligned[i]
        
        # Alligator conditions
        alligator_bullish = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)
        alligator_bearish = (curr_lips < curr_teeth) and (curr_teeth < curr_jaw)
        
        # Volume confirmation: current volume > 2x average
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish + price > 1d EMA50 + volume confirmation
            if (alligator_bullish and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + price < 1d EMA50 + volume confirmation
            elif (alligator_bearish and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price < 1d EMA50
            if (not alligator_bullish) or (curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price > 1d EMA50
            if (not alligator_bearish) or (curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals