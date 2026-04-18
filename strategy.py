#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Supertrend Trend Following with 1d Volume Confirmation and Session Filter (08-20 UTC)
# Uses 4h Supertrend for trend direction, 1h for entry timing, 1d volume filter to avoid low-volume noise.
# Restricts trading to active UTC session (08-20) to reduce false signals.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
# Works in bull markets (follow uptrend) and bear markets (follow downtrend).
name = "1h_Supertrend_4h_1dVolume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = pd.to_datetime(prices['open_time'])
    
    # Pre-calculate session filter (08-20 UTC)
    hours = open_time.hour.values
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Supertrend calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Supertrend on 4h data
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros(len(df_4h))
    direction = np.ones(len(df_4h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_4h)):
        if df_4h['close'].iloc[i] > upper_band[i-1]:
            direction[i] = 1
        elif df_4h['close'].iloc[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and direction to 1h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(volume_filter_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        st_val = supertrend_aligned[i]
        direction_val = direction_aligned[i]
        vol_filter_val = volume_filter_1d_aligned[i]
        
        if position == 0:
            # Long: Uptrend AND price above Supertrend AND volume filter
            if direction_val == 1 and close_val > st_val and vol_filter_val:
                signals[i] = 0.20
                position = 1
            # Short: Downtrend AND price below Supertrend AND volume filter
            elif direction_val == -1 and close_val < st_val and vol_filter_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Downtrend OR price below Supertrend
            if direction_val == -1 or close_val < st_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Uptrend OR price above Supertrend
            if direction_val == 1 or close_val > st_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals