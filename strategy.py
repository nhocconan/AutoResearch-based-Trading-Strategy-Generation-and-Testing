#!/usr/bin/env python3
"""
12-hour Williams Alligator with 1-week EMA trend filter and volume confirmation
Hypothesis: Williams Alligator (Jaws/Teeth/Lips) identifies trending markets. 
Trades taken only when price is aligned with 1-week EMA trend and confirmed by volume spikes.
Designed for ~15-25 trades/year to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_williams_alligator_1w_ema_volume_v1"
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
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    # Jaws: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead  
    # Lips: 5-period SMMA, 3 bars ahead
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean()
        # Initialize first value as SMA
        result = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift for forward displacement (Alligator specific)
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set initial values to NaN after roll
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 1-week EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator lines intertwine (no trend) OR price below 1-week EMA
            lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
            teeth_above_jaws = teeth_shifted[i] > jaws_shifted[i]
            alligator_aligned = lips_above_teeth and teeth_above_jaws
            
            if (not alligator_aligned or close[i] < ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines intertwine (no trend) OR price above 1-week EMA
            lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
            teeth_below_jaws = teeth_shifted[i] < jaws_shifted[i]
            alligator_aligned = lips_below_teeth and teeth_below_jaws
            
            if (not alligator_aligned or close[i] > ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Alligator aligned: Lips > Teeth > Jaws (bullish) OR Lips < Teeth < Jaws (bearish)
            bullish_aligned = (lips_shifted[i] > teeth_shifted[i] and 
                              teeth_shifted[i] > jaws_shifted[i])
            bearish_aligned = (lips_shifted[i] < teeth_shifted[i] and 
                              teeth_shifted[i] < jaws_shifted[i])
            
            # Long: Bullish alignment + price above 1-week EMA + volume spike
            if (bullish_aligned and close[i] > ema_1w_aligned[i] and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bearish alignment + price below 1-week EMA + volume spike
            elif (bearish_aligned and close[i] < ema_1w_aligned[i] and vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals