#!/usr/bin/env python3
"""
6h_ChaikinOscillator_Reversal_1dTrend_Filter
Hypothesis: Chaikin Oscillator identifies accumulation/distribution shifts. Reversal signals when oscillator crosses zero with volume confirmation, filtered by 1d EMA trend. Works in bull/bear by trading reversals against trend exhaustion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Money Flow Multiplier and Volume
    mfm = ((close - low) - (high - close)) / (high - low)
    mfm = np.where((high - low) == 0, 0, mfm)
    mfv = mfm * volume
    
    # Chaikin Oscillator: (3-day EMA of MFV) - (10-day EMA of MFV)
    mfv_series = pd.Series(mfv)
    ema3 = mfv_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = mfv_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    d_uptrend = close > ema_50_1d_aligned
    d_downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup (max of 10, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chaikin[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(d_uptrend[i]) or np.isnan(d_downtrend[i])):
            signals[i] = 0.0
            continue
        
        # Chaikin zero cross with momentum
        chaikin_cross_up = (chaikin[i] > 0) and (chaikin[i-1] <= 0)
        chaikin_cross_down = (chaikin[i] < 0) and (chaikin[i-1] >= 0)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_surge = volume[i] > (vol_ma_20[i] * 1.5) if not np.isnan(vol_ma_20[i]) else False
        
        # Entry conditions
        # Long: Chaikin crosses up from negative + volume surge + not in strong uptrend (fade strength)
        long_entry = chaikin_cross_up and volume_surge and not d_uptrend[i]
        
        # Short: Chaikin crosses down from positive + volume surge + not in strong downtrend (fade weakness)
        short_entry = chaikin_cross_down and volume_surge and not d_downtrend[i]
        
        # Exit when Chaikin crosses zero in opposite direction
        long_exit = chaikin_cross_down
        short_exit = chaikin_cross_up
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ChaikinOscillator_Reversal_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0