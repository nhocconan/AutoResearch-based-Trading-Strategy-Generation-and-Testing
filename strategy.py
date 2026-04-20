#!/usr/bin/env python3
"""
1h_4d_1d_Camarilla_R1S1_Breakout_VolumeFilter_Trend
Hypothesis: On 1h timeframe, trade breakouts of daily Camarilla R1/S1 levels with 4h volume confirmation and 1d trend filter.
Long when price breaks above daily R1 with volume spike and 1d close > EMA50; short when breaks below daily S1 with volume spike and 1d close < EMA50.
Uses higher timeframe (1d) for structural bias and 4h for volume confirmation to reduce false signals.
Designed for 15-30 trades/year to avoid fee drag. Works in bull/bear: breaks indicate momentum continuation, volume confirms validity, trend filter avoids counter-trend whipsaws.
"""

name = "1h_4d_1d_Camarilla_R1S1_Breakout_VolumeFilter_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume average for spike detection (20-period)
    vol_4h = df_4h['volume'].values
    vol_avg_4h = np.full(len(vol_4h), np.nan)
    for i in range(len(vol_4h)):
        if i >= 19:  # 20-period average
            vol_avg_4h[i] = np.mean(vol_4h[i-19:i+1])
    vol_avg_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 168  # Need 7 days of 1h data to ensure prior day exists
    
    for i in range(start_idx, n):
        # Calculate daily Camarilla levels using prior day's OHLC
        # Need to get the previous completed day's data
        # Find index of 1d bar that completed before current 1h bar
        # Since we use align_htf_to_ltf, we can safely use current aligned values
        # but for OHLC we need actual prior day values
        
        # Get prior day's OHLC from 1d data
        # We'll use the last completed 1d bar
        # To avoid look-ahead, we use data up to the prior day
        
        # Calculate how many 1h bars in a day: 24
        # We need to look back at least 24*1=24 hours for prior day, but safer to use more
        
        # Instead, we get the prior day's completed OHLC by indexing into 1d data
        # We need to find which 1d bar corresponds to the day before the current 1h bar's date
        # Since we can't easily do that in loop without look-ahead, we use a different approach:
        # We'll calculate Camarilla levels using the prior day's data that is known to be complete
        
        # Simpler: use the 1d bar that ended at 00:00 UTC of the current day
        # But to avoid look-ahead complexity, we use the fact that align_htf_to_ltf ensures
        # we only use completed 1d bars, so we can use the current aligned 1d values for reference
        # but we need actual OHLC, not just close
        
        # Alternative: calculate the prior day's OHLC by looking back in 1h data
        # Prior day = 24 hours ago = 24 bars back
        if i < 24:
            continue
            
        # Get prior day's OHLC from 1h data (24 hours back to 1 hour back)
        # Actually, we want the full day before today, so 24*2=48 hours back to 24 hours back
        start_idx_day = i - 48
        end_idx_day = i - 24
        if start_idx_day < 0:
            continue
            
        prior_day_high = np.max(prices['high'].iloc[start_idx_day:end_idx_day])
        prior_day_low = np.min(prices['low'].iloc[start_idx_day:end_idx_day])
        prior_day_close = prices['close'].iloc[end_idx_day - 1]  # Last hour of prior day
        
        # Calculate Camarilla levels
        range_val = prior_day_high - prior_day_low
        if range_val <= 0:
            continue
            
        # Camarilla R1 and S1 levels
        r1 = prior_day_close + (range_val * 1.1 / 12)
        s1 = prior_day_close - (range_val * 1.1 / 12)
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Get trend filter: 1d EMA50 aligned
        # For trend, we want to know if the prior day's close was above/below EMA50
        # Use the EMA50 value from the prior day
        # Since we can't easily get prior day's EMA in loop, we use current aligned EMA
        # but that's not ideal. Instead, we use the close vs EMA50 of the prior day
        # We'll approximate by using the EMA50 value from 24 hours ago (prior day close time)
        
        # Get EMA50 value from prior day's close (24 hours ago)
        if i >= 24:
            ema_50_prior_day = ema_50_1d_aligned[i - 24]
        else:
            continue
            
        # Volume spike: current volume > 1.5x 4h average volume
        vol_spike = (not np.isnan(vol_avg_4h_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_4h_aligned[i])
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and prior day close > EMA50
            if current_close > r1 and vol_spike and prior_day_close > ema_50_prior_day:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike and prior day close < EMA50
            elif current_close < s1 and vol_spike and prior_day_close < ema_50_prior_day:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or prior day close < EMA50 (trend change)
            if current_close < s1 or prior_day_close < ema_50_prior_day:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above R1 or prior day close > EMA50 (trend change)
            if current_close > r1 or prior_day_close > ema_50_prior_day:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals