#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips).
# Jaw (Blue): 13-period SMMA shifted 8 bars forward
# Teeth (Red): 8-period SMMA shifted 5 bars forward  
# Lips (Green): 5-period SMMA shifted 3 bars forward
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1w EMA50 AND volume > 2.0x 20-bar average
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1w EMA50 AND volume > 2.0x 20-bar average
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw)
# Williams Alligator identifies trend absence (sleeping Alligator) vs trend formation (awakening Alligator)
# 1w EMA50 filter ensures we only trade in the direction of the dominant weekly trend
# Volume confirmation filters for institutional participation
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components (SMMA = Smoothed Moving Average)
    # SMMA calculation: today's SMMA = (yesterday's SMMA * (period-1) + today's price) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate Alligator lines
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Apply forward shifts (Alligator specific)
    jaw_shifted = np.roll(jaw, 8)   # Jaw shifted 8 bars forward
    teeth_shifted = np.roll(teeth, 5) # Teeth shifted 5 bars forward
    lips_shifted = np.roll(lips, 3)   # Lips shifted 3 bars forward
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
            
            # Long: Bullish alignment, uptrend (price > 1w EMA50), volume confirmation
            if (bullish_alignment and 
                curr_close > ema_50_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment, downtrend (price < 1w EMA50), volume confirmation
            elif (bearish_alignment and 
                  curr_close < ema_50_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Bullish alignment breaks (Lips crosses below Teeth OR Teeth crosses below Jaw)
            bullish_alignment = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
            if not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Bearish alignment breaks (Lips crosses above Teeth OR Teeth crosses above Jaw)
            bearish_alignment = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
            if not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals