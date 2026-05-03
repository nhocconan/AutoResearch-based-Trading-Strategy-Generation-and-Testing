#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivots identify key support/resistance levels where price often reverses or accelerates.
# Breakout above R3 or below S3 with 1d EMA34 trend alignment and volume spike captures strong moves.
# Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year) minimizing fee drag.
# Works in bull markets via R3 breakout continuation and bear markets via S3 breakdown continuation.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Need at least 1 bar for pivot calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels from previous 1d bar
        # Need previous day's high, low, close
        if i < 1:
            continue
            
        # Get 1d data indices for previous completed day
        # We use the 1d data that ends at or before current 12h bar
        prev_day_high = df_1d['high'].iloc[-1] if len(df_1d) > 0 else high[i-1]
        prev_day_low = df_1d['low'].iloc[-1] if len(df_1d) > 0 else low[i-1]
        prev_day_close = df_1d['close'].iloc[-1] if len(df_1d) > 0 else close[i-1]
        
        # Actually, we need to get the proper previous day's OHLC
        # Since we're iterating, we should use rolling window approach for safety
        # But for simplicity and to avoid look-ahead, we use completed 1d bars
        # We'll align the 1d OHLC to 12h timeframe
        
        # Recalculate: get aligned 1d OHLC for previous completed day
        if i >= 1:
            # We need to access 1d data that is already completed
            # Use a safer approach: calculate pivots from aligned 1d series
            
            # Get aligned 1d OHLC (we'll compute these once outside loop for efficiency)
            pass  # We'll handle this differently
        
        # Better approach: calculate Camarilla from 1d data and align
        # But to avoid look-ahead, we use the previous completed 1d bar's data
        
        # For now, use a simplified approach that avoids look-ahead:
        # Calculate typical price and use previous bar's range
        
        # Actually, let's compute the 1d OHLC properly aligned
        # We'll do this outside the loop
        
        # Reset and compute properly
        
    # Re-implement with proper pre-computation
    
    # Pre-compute 1d OHLC aligned to 12h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1d OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align 1d OHLC to 12h timeframe (completed bars only)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA on 12h
    if n >= 20:
        vol_series = pd.Series(volume)
        vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=1).mean().values
    else:
        vol_ema_20 = volume.copy()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar for pivot
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or 
            np.isnan(close_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels from PREVIOUS completed 1d bar
        # Use the 1d bar that ended before current 12h bar
        # Since 1d data is aligned, we use the previous value to avoid look-ahead
        if i >= 1:
            prev_high = high_1d_aligned[i-1]
            prev_low = low_1d_aligned[i-1]
            prev_close = close_1d_aligned[i-1]
        else:
            # Not enough data
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Camarilla R3 and S3 levels
        r3 = prev_close + range_val * 1.1 / 4
        s3 = prev_close - range_val * 1.1 / 4
        
        # Volume confirmation
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 in 1d uptrend with volume spike
            if close[i] > r3 and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in 1d downtrend with volume spike
            elif close[i] < s3 and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below R3 or loses 1d uptrend
            if close[i] < r3 or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above S3 or loses 1d downtrend
            if close[i] > s3 or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals