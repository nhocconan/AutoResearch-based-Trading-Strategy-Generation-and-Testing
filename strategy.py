#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day trend filter and volume confirmation
# Long when Williams Alligator is bullish (jaw < teeth < lips) and price above 1-day EMA50 with volume >1.5x 6-period average
# Short when Williams Alligator is bearish (jaw > teeth > lips) and price below 1-day EMA50 with volume >1.5x 6-period average
# Exit when Alligator lines cross (jaws cross teeth) indicating trend exhaustion
# 1-day EMA50 acts as a trend filter to avoid counter-trend trades
# Williams Alligator (13,8,5 smoothed) captures multi-timeframe momentum
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components (13,8,5 smoothed)
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods
    # Lips: 5-period SMMA smoothed by 3 periods
    def smoothed_mma(data, period):
        """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        # Apply smoothing: SMMA(t) = (SMMA(t-1)*(period-1) + price(t)) / period
        smoothed = np.full_like(data, np.nan, dtype=float)
        smoothed[period-1] = sma[period-1]
        for i in range(period, len(data)):
            if not np.isnan(smoothed[i-1]) and not np.isnan(data[i]):
                smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            else:
                smoothed[i] = np.nan
        return smoothed
    
    # Calculate Alligator lines
    jaw = smoothed_mma(close, 13)  # 13-period
    jaw = smoothed_mma(jaw, 8)     # smoothed by 8
    
    teeth = smoothed_mma(close, 8)   # 8-period
    teeth = smoothed_mma(teeth, 5)   # smoothed by 5
    
    lips = smoothed_mma(close, 5)    # 5-period
    lips = smoothed_mma(lips, 3)     # smoothed by 3
    
    # Calculate 1-day EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 6-period volume average
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    # Align indicators to 6-hour timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need enough for smoothing)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        # Williams Alligator conditions
        bullish_alligator = (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i])
        bearish_alligator = (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i])
        
        if position == 0:
            # Long setup: bullish Alligator + price above 1-day EMA50 + volume confirmation
            if (bullish_alligator and 
                price > ema_50_1d_aligned[i] and 
                vol_current > 1.5 * vol_ma_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: bearish Alligator + price below 1-day EMA50 + volume confirmation
            elif (bearish_alligator and 
                  price < ema_50_1d_aligned[i] and 
                  vol_current > 1.5 * vol_ma_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator loses bullish alignment (jaws cross teeth)
            if jaw_aligned[i] > teeth_aligned[i]:  # Jaw crossed above teeth
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator loses bearish alignment (jaws cross teeth)
            if jaw_aligned[i] < teeth_aligned[i]:  # Jaw crossed below teeth
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0