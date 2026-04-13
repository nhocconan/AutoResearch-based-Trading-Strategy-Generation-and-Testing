#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1w/1d regime filter + volume confirmation
    # Long when: price > Alligator Jaw (13,8,5) AND weekly close > daily EMA50 AND volume > 1.5x avg
    # Short when: price < Alligator Jaw AND weekly close < daily EMA50 AND volume > 1.5x avg
    # Exit when: price crosses Alligator Teeth (8-period smoothed median)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Alligator identifies trend; weekly/daily alignment filters counter-trend noise; volume confirms.
    # Works in bull (trend-following with alignment) and bear (strong aligned moves only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Williams Alligator (13,8,5) - SMMA (Smoothed Moving Average)
    def smma(source, length):
        if length < 1:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is SMA
        if len(source) >= length:
            result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
        for i in range(length, len(source)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma((high_12h + low_12h) / 2, 13)  # Jaw: 13-period SMMA of median price
    teeth = smma((high_12h + low_12h) / 2, 8)   # Teeth: 8-period SMMA
    lips = smma((high_12h + low_12h) / 2, 5)    # Lips: 5-period SMMA
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for regime filter (daily EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly close > weekly EMA20 for bullish weekly regime
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_bullish = close_1w > ema_20_1w  # Bullish when price above weekly EMA20
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-bar average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Warmup period: need enough data for Alligator (13-period SMMA needs 13 bars)
    warmup = max(20, 13)  # 20 for volume, 13 for Alligator
    
    for i in range(warmup, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Median price for Alligator comparison
        median_price = (high[i] + low[i]) / 2.0
        
        # Alligator alignment: all three lines aligned in same direction
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        alligator_bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        alligator_bearish = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Weekly regime: bullish when price > weekly EMA20
        weekly_regime_bullish = weekly_bullish_aligned[i] > 0.5
        
        # Daily regime: bullish when price > daily EMA50
        daily_regime_bullish = close[i] > ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = (alligator_bullish and 
                     weekly_regime_bullish and 
                     daily_regime_bullish and 
                     volume_confirmed[i] and 
                     position != 1)
        
        short_entry = (alligator_bearish and 
                      (not weekly_regime_bullish) and 
                      (not daily_regime_bullish) and 
                      volume_confirmed[i] and 
                      position != -1)
        
        # Exit conditions: price crosses Alligator Teeth (8-period)
        exit_long = (position == 1 and median_price < teeth_aligned[i])
        exit_short = (position == -1 and median_price > teeth_aligned[i])
        
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

name = "12h_1w_1d_alligator_regime_volume_v1"
timeframe = "12h"
leverage = 1.0