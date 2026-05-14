#!/usr/bin/env python3
# 4h_williams_alligator_1d_trend_volume_v1
# Hypothesis: Williams Alligator on 4h (Jaws=13, Teeth=8, Lips=5) with 1d EMA(50) trend filter and volume confirmation.
# In bull markets: Go long when Lips cross above Teeth and Jaws (bullish alignment) with volume > 1.5x avg and price above 1d EMA.
# In bear markets: Go short when Lips cross below Teeth and Jaws (bearish alignment) with volume > 1.5x avg and price below 1d EMA.
# Uses Williams Alligator to capture trend changes and avoids choppy markets via strict alignment conditions.
# Target: 20-40 trades/year to avoid fee decay while capturing sustained moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_williams_alligator_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h: SMMA (Smoothed Moving Average)
    # Jaws: SMA(13) shifted 8 bars forward
    # Teeth: SMA(8) shifted 5 bars forward  
    # Lips: SMA(5) shifted 3 bars forward
    def smma(data, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaws = smma(close, 13)  # Blue line
    teeth = smma(close, 8)   # Red line
    lips = smma(close, 5)    # Green line
    
    # Shift the lines as per Williams Alligator specification
    jaws_shifted = np.roll(jaws, 8)   # Shift 8 bars forward
    teeth_shifted = np.roll(teeth, 5) # Shift 5 bars forward
    lips_shifted = np.roll(lips, 3)   # Shift 3 bars forward
    
    # Set NaN for shifted positions that don't have data
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 13 + 8)  # Ensure we have data for shifted jaws
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaws_shifted[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bearish Alligator alignment (Lips < Teeth < Jaws) or opposite signal with volume
            bearish_alignment = (lips_shifted[i] < teeth_shifted[i] < jaws_shifted[i])
            opposite_signal = (lips_shifted[i] < teeth_shifted[i] and 
                             volume[i] > 1.5 * avg_volume[i] and 
                             close[i] < ema_50_1d_aligned[i])
            if bearish_alignment or opposite_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment (Lips > Teeth > Jaws) or opposite signal with volume
            bullish_alignment = (lips_shifted[i] > teeth_shifted[i] > jaws_shifted[i])
            opposite_signal = (lips_shifted[i] > teeth_shifted[i] and 
                             volume[i] > 1.5 * avg_volume[i] and 
                             close[i] > ema_50_1d_aligned[i])
            if bullish_alignment or opposite_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Bullish Alligator alignment: Lips > Teeth > Jaws
            bullish_alignment = (lips_shifted[i] > teeth_shifted[i] > jaws_shifted[i])
            # Bearish Alligator alignment: Lips < Teeth < Jaws  
            bearish_alignment = (lips_shifted[i] < teeth_shifted[i] < jaws_shifted[i])
            
            # Long entry: Bullish alignment with volume and 1d uptrend
            if bullish_alignment and volume_ok and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish alignment with volume and 1d downtrend
            elif bearish_alignment and volume_ok and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals