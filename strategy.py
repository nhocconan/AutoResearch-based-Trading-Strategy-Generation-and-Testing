#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift).
# Trend up when Lips > Teeth > Jaw, trend down when Lips < Teeth < Jaw.
# Combined with 1d EMA50 trend filter and volume > 1.5x 20-period average for confirmation.
# Designed for low-frequency trading (12h) to minimize fees while capturing major trends.
name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (using 12h data)
    # Jaw: 13-period SMMA, 8 bars into the future
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift forward 8 bars
    
    # Teeth: 8-period SMMA, 5 bars into the future
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift forward 5 bars
    
    # Lips: 5-period SMMA, 3 bars into the future
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift forward 3 bars
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h[i]) or np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or 
            np.isnan(lips_vals[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator trend conditions
        trend_up = lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i]
        trend_down = lips_vals[i] < teeth_vals[i] and teeth_vals[i] < jaw_vals[i]
        
        if position == 0:
            # Long: Alligator bullish alignment + 1d uptrend + volume confirmation
            if trend_up and close[i] > ema_50_12h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment + 1d downtrend + volume confirmation
            elif trend_down and close[i] < ema_50_12h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator bearish alignment or 1d trend reversal
            if not trend_up or close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator bullish alignment or 1d trend reversal
            if not trend_down or close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals