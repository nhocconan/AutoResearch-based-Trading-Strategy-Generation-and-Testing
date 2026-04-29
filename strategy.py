#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# Long: Lips > Teeth > Jaw AND price > 1d EMA50 AND volume > 1.5x 20-bar avg (bullish alignment)
# Short: Lips < Teeth < Jaw AND price < 1d EMA50 AND volume > 1.5x 20-bar avg (bearish alignment)
# Exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw) OR volume drops below average
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
# Alligator catches trends early; volume filter avoids false breakouts; 1d EMA ensures higher-timeframe trend alignment

name = "4h_Williams_Alligator_1dEMA50_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (using smoothed moving average - SMMA)
    # SMMA is similar to EMA but with different smoothing: SMMA(i) = (SMMA(i-1)*(n-1) + close(i)) / n
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1)*(period-1) + data(i)) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw = smma(close, 13)  # Jaw: 13-period SMMA
    teeth = smma(close, 8)   # Teeth: 8-period SMMA
    lips = smma(close, 5)    # Lips: 5-period SMMA
    
    # Apply shifts (Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set shifted values to NaN for the shifted periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align HTF indicators to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Alligator
    
    for i in range(start_idx, n):
        # Skip if any Alligator value is NaN
        if np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator relationships
        lips_above_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_above_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_below_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_below_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Alligator lines cross (bullish alignment broken) OR volume drops below average
            if not bullish_alignment or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (bearish alignment broken) OR volume drops below average
            if not bearish_alignment or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish Alligator alignment AND price > 1d EMA50 AND volume confirmation
            if (bullish_alignment and 
                curr_close > curr_ema_1d and
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator alignment AND price < 1d EMA50 AND volume confirmation
            elif (bearish_alignment and 
                  curr_close < curr_ema_1d and
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals