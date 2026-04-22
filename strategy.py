#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator system with 1d trend filter
    # Williams Alligator uses smoothed medians (Jaw/Teeth/Lips) to identify trends
    # Jaw (13-period smoothed median) = trend direction
    # Teeth (8-period smoothed median) = entry signal
    # Lips (5-period smoothed median) = momentum confirmation
    # 1d EMA50 filter ensures alignment with higher timeframe trend
    # Volume confirmation (1.5x 20-period MA) filters false signals
    # Target: 15-25 trades/year with high win rate in both bull and bear markets
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components (using median prices)
    median_price = (high + low) / 2
    
    # Jaw: 13-period smoothed median (8 periods shift)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).median().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period smoothed median (5 periods shift)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).median().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period smoothed median (3 periods shift)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).median().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips above Teeth above Jaw (bullish alignment) + volume confirmation + price above 1d EMA50
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_confirm[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth below Jaw (bearish alignment) + volume confirmation + price below 1d EMA50
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_confirm[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to balance (Lips cross Teeth) or trend reversal
            if position == 1:
                if lips[i] < teeth[i]:  # Lips cross below Teeth - exit long
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i]:  # Lips cross above Teeth - exit short
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0