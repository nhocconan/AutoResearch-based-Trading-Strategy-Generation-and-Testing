#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with volume confirmation and 1d trend filter
# Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength
# Price above all three lines = uptrend, below all three = downtrend
# Volume > 1.3x average confirms trend strength
# 1d EMA50 filter ensures alignment with higher timeframe trend
# Low turnover expected: ~15-25 trades/year per symbol
# Works in bull markets (catching uptrends) and bear markets (catching downtrends)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams Alligator and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator (13,8,5 smoothed with 8,5,3)
    # Jaw (13-period SMMA, 8-period shift)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw_raw[:-8]]) if len(jaw_raw) > 8 else np.full_like(close, np.nan)
    
    # Teeth (8-period SMMA, 5-period shift)
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth_raw[:-5]]) if len(teeth_raw) > 5 else np.full_like(close, np.nan)
    
    # Lips (5-period SMMA, 3-period shift)
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips_raw[:-3]]) if len(lips_raw) > 3 else np.full_like(close, np.nan)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: price above all three = uptrend, below all three = downtrend
        bullish_alignment = (close[i] > jaw_aligned[i] and 
                            close[i] > teeth_aligned[i] and 
                            close[i] > lips_aligned[i])
        bearish_alignment = (close[i] < jaw_aligned[i] and 
                            close[i] < teeth_aligned[i] and 
                            close[i] < lips_aligned[i])
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # 1d EMA50 trend filter: price above EMA50 = bullish bias, below = bearish bias
        price_above_ema = close[i] > ema_50_aligned[i]
        price_below_ema = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Enter long: bullish alignment + volume + price above 1d EMA50
            if (bullish_alignment and 
                volume_confirmed and 
                price_above_ema):
                position = 1
                signals[i] = position_size
            # Enter short: bearish alignment + volume + price below 1d EMA50
            elif (bearish_alignment and 
                  volume_confirmed and 
                  price_below_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Jaw line (trend change)
            if close[i] < jaw_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Jaw line (trend change)
            if close[i] > jaw_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_WilliamsAlligator_Volume_EMA50_v1"
timeframe = "12h"
leverage = 1.0