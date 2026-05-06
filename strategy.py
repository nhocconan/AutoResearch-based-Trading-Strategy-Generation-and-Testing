#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator (Jaw/Teeth/Lips) with volume confirmation
# Long when price > Alligator Lips AND Lips > Teeth AND Teeth > Jaw (bullish alignment) AND volume > 1.5 * avg_volume(34)
# Short when price < Alligator Lips AND Lips < Teeth AND Teeth < Jaw (bearish alignment) AND volume > 1.5 * avg_volume(34)
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams Alligator identifies trend direction via smoothed medians; filters choppy markets
# Volume confirmation ensures institutional participation in trending moves
# Works in bull (continuation uptrends) and bear (continuation downtrends) by following Alligator alignment

name = "4h_1dWilliamsAlligator_Alignment_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for SMMA(13,8,5)
        return np.zeros(n)
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator: three smoothed medians (SMMA)
    # Jaw: SMMA(median, 13, 8) - blue line
    # Teeth: SMMA(median, 8, 5) - red line
    # Lips: SMMA(median, 5, 3) - green line
    def smma(values, period, prev=None):
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        sma = np.mean(values[:period])
        result[period-1] = sma
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(median_1d, 13, 8)
    teeth = smma(median_1d, 8, 5)
    lips = smma(median_1d, 5, 3)
    
    # Align 1d Alligator lines to 4h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate volume confirmation: volume > 1.5 * 34-period average volume on 4h
    avg_volume_34 = pd.Series(volume).rolling(window=34, min_periods=34).mean().values
    volume_confirm = volume > (1.5 * avg_volume_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(avg_volume_34[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long: bullish Alligator alignment with volume confirmation
            if bullish_alignment and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment with volume confirmation
            elif bearish_alignment and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bullish alignment breaks
            if not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bearish alignment breaks
            if not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals