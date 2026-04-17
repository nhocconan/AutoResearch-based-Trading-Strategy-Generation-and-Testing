#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (13,8,5 SMAs)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) SMAs
    jaw_1d = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    # Align to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need Williams Alligator, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or
            np.isnan(ema50_12h[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish; Lips < Teeth < Jaw = bearish
        bullish_alignment = (lips_12h[i] > teeth_12h[i]) and (teeth_12h[i] > jaw_12h[i])
        bearish_alignment = (lips_12h[i] < teeth_12h[i]) and (teeth_12h[i] < jaw_12h[i])
        
        # Trend filter: price relative to weekly EMA50
        price_above_weekly_ema = close[i] > ema50_12h[i]
        price_below_weekly_ema = close[i] < ema50_12h[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment + price above weekly EMA + volume
            if bullish_alignment and price_above_weekly_ema and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + price below weekly EMA + volume
            elif bearish_alignment and price_below_weekly_ema and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish alignment OR price crosses below weekly EMA
            if bearish_alignment or (close[i] < ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish alignment OR price crosses above weekly EMA
            if bullish_alignment or (close[i] > ema50_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_EMA50_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0