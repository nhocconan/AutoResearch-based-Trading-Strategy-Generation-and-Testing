#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips).
# Long when Lips > Teeth > Jaw (bullish alignment) with price above Lips, 1d uptrend, and volume > 1.5x 20-bar avg.
# Short when Lips < Teeth < Jaw (bearish alignment) with price below Lips, 1d downtrend, and volume > 1.5x 20-bar avg.
# Exit when Alligator alignment breaks or price crosses the Teeth line.
# Uses 1d EMA50 for longer-term trend filter to reduce false signals in choppy markets.
# Williams Alligator is effective in both trending and ranging markets, providing clear trend direction signals.
# Timeframe: 4h, HTF: 1d as per experiment guidelines.

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator parameters (4h timeframe)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate smoothed moving averages (SMMA) for Alligator
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=np.float64)
        result = np.full_like(data, np.nan, dtype=np.float64)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Apply shifts (Alligator lines are shifted into the future)
    jaw_shifted = np.roll(jaw, jaw_shift)
    teeth_shifted = np.roll(teeth, teeth_shift)
    lips_shifted = np.roll(lips, lips_shift)
    
    # Set NaN for shifted values that would look ahead
    jaw_shifted[:jaw_shift] = np.nan
    teeth_shifted[:teeth_shift] = np.nan
    lips_shifted[:lips_shift] = np.nan
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_shift, teeth_shift, lips_shift, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_lips = lips_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_jaw = jaw_shifted[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Alligator alignment conditions
        bullish_alignment = curr_lips > curr_teeth > curr_jaw
        bearish_alignment = curr_lips < curr_teeth < curr_jaw
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish alignment, price above Lips, 1d uptrend, volume confirmation
            if (bullish_alignment and 
                curr_close > curr_lips and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price below Lips, 1d downtrend, volume confirmation
            elif (bearish_alignment and 
                  curr_close < curr_lips and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: alignment breaks or price crosses Teeth
            if not bullish_alignment or curr_close < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: alignment breaks or price crosses Teeth
            if not bearish_alignment or curr_close > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals