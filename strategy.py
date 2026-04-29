#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend + volume confirmation
# Williams Alligator uses smoothed medians (Jaw/Teeth/Lips) to identify trends
# Jaw (13-period smoothed median), Teeth (8-period), Lips (5-period)
# Trend filter: price > 1d EMA34 for longs, price < 1d EMA34 for shorts
# Volume confirmation: current volume > 2.0x 20-period average
# Exit: Alligator lines converge (Lips crosses Teeth/Jaw) or opposite signal
# Designed for ~15-25 trades/year on 12h timeframe to minimize fee drag
# Works in bull/bear via trend filter and Alligator convergence exits

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Smoothed medians (using SMMA - Smoothed Moving Average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5)
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift to avoid look-ahead (use previous bar values)
    jaw_shifted = np.roll(jaw, 1)
    teeth_shifted = np.roll(teeth, 1)
    lips_shifted = np.roll(lips, 1)
    jaw_shifted[0] = np.nan
    teeth_shifted[0] = np.nan
    lips_shifted[0] = np.nan
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 13)  # Volume, 1d EMA34, and Alligator warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Check for Alligator convergence (exit condition)
        # Convergence when Lips is between Jaw and Teeth
        lips_between = (curr_lips > min(curr_jaw, curr_teeth) and curr_lips < max(curr_jaw, curr_teeth))
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator convergence OR price below 1d EMA34
            if lips_between or curr_close < curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator convergence OR price above 1d EMA34
            if lips_between or curr_close > curr_ema34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Alligator alignment: Lips > Teeth > Jaw for uptrend, Lips < Teeth < Jaw for downtrend
            lips_above_teeth = curr_lips > curr_teeth
            teeth_above_jaw = curr_teeth > curr_jaw
            lips_below_teeth = curr_lips < curr_teeth
            teeth_below_jaw = curr_teeth < curr_jaw
            
            # Long when Alligator aligned up, 1d EMA34 up-trend, volume confirmed
            if lips_above_teeth and teeth_above_jaw and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when Alligator aligned down, 1d EMA34 down-trend, volume confirmed
            elif lips_below_teeth and teeth_below_jaw and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals