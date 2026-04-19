#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with weekly trend filter and daily volume confirmation.
# Long when price > Alligator Jaw (teeth > lips) AND weekly close > weekly SMA(50) AND daily volume > 1.5x daily average volume
# Short when price < Alligator Jaw (teeth < lips) AND weekly close < weekly SMA(50) AND daily volume > 1.5x daily average volume
# Exit when price crosses back through Alligator Jaw
# Uses Alligator for trend identification, weekly trend filter for multi-timeframe alignment, volume for confirmation.
# Target: 15-25 trades/year per symbol.
name = "12h_Alligator_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly SMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_sma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    # Calculate daily average volume (20-period)
    daily_volume = df_1d['volume'].values
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_ma_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_ma)
    
    # Calculate Williams Alligator (13,8,5) smoothed with SMA
    # Jaw (13-period, shifted 8 bars forward)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)
    # Teeth (8-period, shifted 5 bars forward)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)
    # Lips (5-period, shifted 3 bars forward)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13+8, 8+5, 5+3, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_sma50_aligned[i]) or np.isnan(daily_vol_ma_aligned[i]) or
            np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_sma50_val = weekly_sma50_aligned[i]
        daily_vol_ma_val = daily_vol_ma_aligned[i]
        vol = volume[i]
        jaw_val = jaw_values[i]
        teeth_val = teeth_values[i]
        lips_val = lips_values[i]
        
        # Alligator condition: teeth > lips for bullish alignment, teeth < lips for bearish
        alligator_bull = teeth_val > lips_val
        alligator_bear = teeth_val < lips_val
        
        # Weekly trend filter: price above/below weekly SMA(50)
        weekly_uptrend = close[i] > weekly_sma50_val
        weekly_downtrend = close[i] < weekly_sma50_val
        
        # Volume confirmation: current volume > 1.5x daily average volume
        volume_confirm = vol > 1.5 * daily_vol_ma_val
        
        if position == 0:
            # Long entry: price > jaw AND teeth > lips AND weekly uptrend AND volume confirmation
            if price > jaw_val and alligator_bull and weekly_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < jaw AND teeth < lips AND weekly downtrend AND volume confirmation
            elif price < jaw_val and alligator_bear and weekly_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below jaw
            if price < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above jaw
            if price > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals