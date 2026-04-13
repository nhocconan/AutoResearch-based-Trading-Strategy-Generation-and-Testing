#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams Alligator trend with 1w HTF filter
    # Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
    # Trend up: Lips > Teeth > Jaw; Trend down: Lips < Teeth < Jaw
    # Filter: Only trade when 1w close > 1w EMA200 (bull) or < EMA200 (bear)
    # Enter on Alligator alignment in trend direction, exit on reversal
    # Works in bull (continuation) and bear (counter-trend at extremes)
    # Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Get 1d data for primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_price_1d = (high_1d + low_1d) / 2
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 1d median price
    # Jaw: 13-period SMA, shifted 8 bars
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 1d timeframe (already aligned via get_htf_data)
    # No additional alignment needed as we're using 1d data directly
    
    # 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(lips_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(jaw_1d[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        lips_above_teeth = lips_1d[i] > teeth_1d[i]
        teeth_above_jaw = teeth_1d[i] > jaw_1d[i]
        lips_below_teeth = lips_1d[i] < teeth_1d[i]
        teeth_below_jaw = teeth_1d[i] < jaw_1d[i]
        
        alligator_up = lips_above_teeth and teeth_above_jaw
        alligator_down = lips_below_teeth and teeth_below_jaw
        
        # 1w trend filter: only trade in direction of 1w trend
        close_1w_now = close_1d[i]  # approximate 1w close using current 1d close (will be aligned properly)
        # Actually get the aligned 1w close for proper comparison
        df_1w_close = get_htf_data(prices, '1w')['close'].values
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w_close)
        
        # Since we already have ema_200_1w_aligned, we need the actual 1w close aligned
        # Re-get the 1w close series for alignment
        close_1w_series = df_1w['close'].values
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w_series)
        
        # 1w trend: bullish if price > EMA200, bearish if price < EMA200
        bullish_1w = close_1w_aligned[i] > ema_200_1w_aligned[i]
        bearish_1w = close_1w_aligned[i] < ema_200_1w_aligned[i]
        
        # Entry conditions
        long_entry = alligator_up and bullish_1w and position != 1
        short_entry = alligator_down and bearish_1w and position != -1
        
        # Exit conditions: reverse Alligator alignment
        exit_long = position == 1 and (lips_1d[i] < teeth_1d[i] or teeth_1d[i] < jaw_1d[i])
        exit_short = position == -1 and (lips_1d[i] > teeth_1d[i] or teeth_1d[i] > jaw_1d[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williams_alligator_trend_filter_v1"
timeframe = "1d"
leverage = 1.0