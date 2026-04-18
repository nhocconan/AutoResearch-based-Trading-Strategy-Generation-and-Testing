#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with daily EMA50 filter and volume confirmation.
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# Price above all three lines indicates uptrend, below indicates downtrend.
# Daily EMA50 ensures alignment with longer-term trend, avoiding counter-trend trades.
# Volume confirmation adds conviction to entries.
# Designed for low trade frequency (12-37/year) in 12h timeframe to minimize fee drag.
# Works in bull markets (price above alligator with EMA50 up) and bear markets 
# (price below alligator with EMA50 down).
name = "12h_WilliamsAlligator_DailyEMA50_Volume"
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
    
    # Get daily data for Williams Alligator and EMA50 (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator components (using previous day's data to avoid look-ahead)
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA shifted 3 bars
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Williams Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_raw)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price above all three Alligator lines AND above daily EMA50 AND volume
            price_above_jaw = close[i] > jaw_aligned[i]
            price_above_teeth = close[i] > teeth_aligned[i]
            price_above_lips = close[i] > lips_aligned[i]
            above_ema50 = close[i] > ema_50_aligned[i]
            
            if vol_confirm and price_above_jaw and price_above_teeth and price_above_lips and above_ema50:
                signals[i] = 0.25
                position = 1
            # Short: price below all three Alligator lines AND below daily EMA50 AND volume
            elif (vol_confirm and 
                  close[i] < jaw_aligned[i] and 
                  close[i] < teeth_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below any Alligator line OR below daily EMA50
            price_below_jaw = close[i] < jaw_aligned[i]
            price_below_teeth = close[i] < teeth_aligned[i]
            price_below_lips = close[i] < lips_aligned[i]
            below_ema50 = close[i] < ema_50_aligned[i]
            
            if price_below_jaw or price_below_teeth or price_below_lips or below_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above any Alligator line OR above daily EMA50
            price_above_jaw = close[i] > jaw_aligned[i]
            price_above_teeth = close[i] > teeth_aligned[i]
            price_above_lips = close[i] > lips_aligned[i]
            above_ema50 = close[i] > ema_50_aligned[i]
            
            if price_above_jaw or price_above_teeth or price_above_lips or above_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals