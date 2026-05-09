#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and entry points.
# Combines with 1d EMA50 for higher timeframe trend alignment and volume confirmation
# to reduce false signals. Designed for 12h timeframe with target of 50-150 trades
# over 4 years (12-37/year). Works in bull/bear markets by requiring alignment
# between Alligator signals and higher timeframe trend.
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
    
    # Calculate Williams Alligator on 12h data
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMAs for Alligator lines
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift).values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # Align Alligator lines (already calculated on 12h data, no additional alignment needed)
    # But we need to ensure we use previous bar values to avoid look-ahead
    jaw_lag = np.roll(jaw, 1)
    teeth_lag = np.roll(teeth, 1)
    lips_lag = np.roll(lips, 1)
    jaw_lag[0] = np.nan
    teeth_lag[0] = np.nan
    lips_lag[0] = np.nan
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_period, teeth_period, lips_period) + max(jaw_shift, teeth_shift, lips_shift) + 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_lag[i]) or np.isnan(teeth_lag[i]) or np.isnan(lips_lag[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_lag[i] > teeth_lag[i] and teeth_lag[i] > jaw_lag[i]
        alligator_short = lips_lag[i] < teeth_lag[i] and teeth_lag[i] < jaw_lag[i]
        
        trend_up = close[i] > ema_50_12h[i]
        trend_down = close[i] < ema_50_12h[i]
        
        if position == 0:
            # Long: Alligator uptrend + price above teeth + uptrend + volume confirmation
            if alligator_long and close[i] > teeth_lag[i] and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + price below teeth + downtrend + volume confirmation
            elif alligator_short and close[i] < teeth_lag[i] and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator death cross (Lips < Jaw) or trend reversal
            if lips_lag[i] < jaw_lag[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator golden cross (Lips > Jaw) or trend reversal
            if lips_lag[i] > jaw_lag[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals