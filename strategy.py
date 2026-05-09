#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw, Teeth, Lips) identifies trend direction and entry opportunities.
# EMA50 on 1d filters for higher timeframe trend alignment, reducing whipsaws.
# Volume > 1.5x 20-period average confirms institutional participation.
# Works in bull/bear markets by requiring trend alignment and clear Alligator signals.
# Target: 50-150 trades over 4 years on 4h timeframe.
name = "4h_WilliamsAlligator_1dEMA50_Trend_Volume"
timeframe = "4h"
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
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 4h data
    # Jaw (13-period SMMA, shifted 8 bars forward)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, shifted 5 bars forward)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, shifted 3 bars forward)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        trend_up = close[i] > ema_50_4h[i]
        trend_down = close[i] < ema_50_4h[i]
        
        if position == 0:
            # Long: Alligator uptrend + price above 1d EMA50 + volume confirmation
            if alligator_up and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + price below 1d EMA50 + volume confirmation
            elif alligator_down and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns down or price crosses below 1d EMA50
            if not alligator_up or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns up or price crosses above 1d EMA50
            if not alligator_down or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals