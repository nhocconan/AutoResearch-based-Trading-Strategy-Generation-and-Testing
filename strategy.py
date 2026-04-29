#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends
# Jaw (13-period SMMA, 8-period shift) = blue line, Teeth (8-period SMMA, 5-period shift) = red line, Lips (5-period SMMA, 3-period shift) = green line
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA34 + volume > 2.0x 20-period average
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA34 + volume > 2.0x 20-period average
# Works in trending markets via Alligator alignment and in ranging markets via volume spike mean reversion
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator components (SMMA = Smoothed Moving Average)
    # SMMA formula: SMMA(i) = (SMMA(i-1) * (period-1) + close(i)) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values are smoothed
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines with standard parameters
    jaw = smma(close, 13)  # Jaw (Blue) - 13-period SMMA
    teeth = smma(close, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(close, 5)   # Lips (Green) - 5-period SMMA
    
    # Apply shifts (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from beginning
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 34, 20)  # Jaw(13+8), EMA34, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if curr_lips <= curr_teeth or curr_teeth <= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if curr_lips >= curr_teeth or curr_teeth >= curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish alignment (Lips > Teeth > Jaw) + uptrend + volume confirmation
            if (curr_lips > curr_teeth and curr_teeth > curr_jaw and 
                curr_close > curr_ema_1d and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment (Lips < Teeth < Jaw) + downtrend + volume confirmation
            elif (curr_lips < curr_teeth and curr_teeth < curr_jaw and 
                  curr_close < curr_ema_1d and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals