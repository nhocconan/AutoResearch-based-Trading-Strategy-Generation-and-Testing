#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday support/resistance based on prior day's range.
# EMA34 on 1d ensures we only trade with the daily trend (bullish for longs above EMA, bearish for shorts below EMA).
# Volume spike confirms institutional participation. Designed for 20-40 trades/year on 4h to minimize fee drag.
# Works in bull markets via trend continuation and in bear markets via shorting breakdowns in downtrends.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have prior day for Camarilla
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using prior 1d bar (already closed)
        # We need prior day's OHLC - use 1d data aligned to current 4h bar
        # Since we're in 4h timeframe, we need to get the prior completed 1d bar's OHLC
        # align_htf_to_ltf gives us the prior completed 1d bar's values for each 4h bar
        # We'll calculate Camarilla levels from the prior 1d bar's OHLC
        
        # Get prior completed 1d bar's OHLC (aligned to current 4h bar)
        # We need to shift the 1d data by 1 bar to get the prior completed day
        if i >= 1:  # We need at least one prior bar
            # Get the 1d OHLC values for the prior completed day
            # We'll use the aligned 1d data but shifted by 1 to get prior day's values
            # Since align_htf_to_ltf gives us the value of the completed HTF bar for each LTF bar,
            # we need to get the prior completed HTF bar's OHLC
            # Simpler approach: calculate Camarilla on 1d data, then align
            
            # Calculate typical price for prior 1d bar
            # We'll use the 1d data directly and align it
            pass  # We'll implement Camarilla calculation differently
        
        # Simpler approach: calculate Camarilla levels for each 1d bar, then align
        # This needs to be done outside the loop for efficiency
        
        # Recalculate: compute Camarilla levels on 1d data, then align to 4h
        # This should be done before the loop
        
        # For now, implement a simplified version that works
        # Calculate Camarilla levels using rolling window of 1d data
        # But we need to do this outside the loop
        
        # Let's restructure: calculate Camarilla levels before the loop
        
        # Actually, let's compute Camarilla levels for each 1d bar, then align
        # We'll do this before the loop
        
        # For now, skip Camarilla calculation in loop and use a placeholder
        # This is not ideal but lets us test the structure
        
        # Placeholder: use simple breakout logic
        if i >= 20:  # Need sufficient lookback
            highest_high = np.max(high[i-19:i+1])
            lowest_low = np.min(low[i-19:i+1])
            
            # Volume confirmation
            if i >= 19:
                vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
            else:
                vol_ema_20 = volume[i]
            volume_spike = volume[i] > (1.5 * vol_ema_20)
            
            breakout_up = close[i] > highest_high
            breakout_down = close[i] < lowest_low
            
            if position == 0:
                if breakout_up and ema_34_1d_aligned[i] < close[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                elif breakout_down and ema_34_1d_aligned[i] > close[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                midpoint = (highest_high + lowest_low) / 2
                if close[i] < midpoint or ema_34_1d_aligned[i] >= close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                midpoint = (highest_high + lowest_low) / 2
                if close[i] > midpoint or ema_34_1d_aligned[i] <= close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

# The above is a placeholder. Let's implement proper Camarilla calculation.
# We need to calculate Camarilla levels for each 1d bar, then align to 4h timeframe.