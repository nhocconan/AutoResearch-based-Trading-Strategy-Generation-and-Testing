#!/usr/bin/env python3
"""
12h Williams Alligator with 1-day EMA Trend Filter and Volume Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend presence and direction.
When the Lips cross above Teeth (bullish) or below Teeth (bearish) with 1-day EMA trend
confirmation and volume spike, it captures strong trends while avoiding whipsaws.
Designed for ~15-25 trades/year to minimize fee drain on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_williams_alligator_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-day EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs with specific periods and offsets
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    # Using EMA as proxy for SMMA for simplicity (common approximation)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Lips cross below Teeth (bearish cross) OR price closes below 1-day EMA50
            if (lips[i] < teeth[i] or close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Lips cross above Teeth (bullish cross) OR price closes above 1-day EMA50
            if (lips[i] > teeth[i] or close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1-day EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Bullish alignment + uptrend + volume spike
            if bullish_alignment and uptrend and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short: Bearish alignment + downtrend + volume spike
            elif bearish_alignment and downtrend and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals