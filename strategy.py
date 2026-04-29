#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Weekly Trend Filter and Volume Confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence and direction
# Weekly trend filter (price vs 21 EMA) ensures alignment with higher timeframe trend
# Volume confirmation validates breakout strength during trending periods
# Works in all regimes: catches trends early and avoids choppy markets
# Target: 12-37 trades/year (50-150 total over 4 years)

name = "12h_WilliamsAlligator_WeeklyTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA21 for trend filter (Alligator Jaw equivalent)
    close_1w = pd.Series(df_1w['close'].values)
    ema21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Williams Alligator components on 12h data
    # Jaw: 13-period SMMA shifted 8 bars
    close_s = pd.Series(close)
    jaw_raw = close_s.rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_raw = close_s.rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips: 5-period SMMA shifted 3 bars
    lips_raw = close_s.rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips[:3] = np.nan  # first 3 values invalid
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 21, 20, 13, 8, 5)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema21_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema21_1w = ema21_1w_aligned[i]
        
        # Determine trend regime from weekly EMA21
        bullish_regime = curr_close > curr_ema21_1w
        bearish_regime = curr_close < curr_ema21_1w
        
        # Alligator signals: 
        # Bullish alignment: Lips > Teeth > Jaw (alligator mouth open upward)
        # Bearish alignment: Lips < Teeth < Jaw (alligator mouth open downward)
        bullish_alligator = (curr_lips > curr_teeth) and (curr_teeth > curr_jaw)
        bearish_alligator = (curr_lips < curr_teeth) and (curr_teeth < curr_jaw)
        
        if position == 0:  # Flat - look for new entries
            # Enter only when Alligator shows clear trend AND volume confirms
            if bullish_alligator and bullish_regime and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            elif bearish_alligator and bearish_regime and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when trend weakens
            # Exit when Alligator alignment breaks (mouth closes) OR reverse signal
            if not bullish_alligator or bearish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when trend weakens
            # Exit when Alligator alignment breaks (mouth closes) OR reverse signal
            if not bearish_alligator or bullish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals