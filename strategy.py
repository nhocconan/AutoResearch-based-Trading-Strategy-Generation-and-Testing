#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d volume confirmation + 1w trend filter
# The Alligator (three SMAs) acts as a trend filter: when jaws (13) > teeth (8) > lips (5) = bullish trend,
# and reverse = bearish trend. In strong trends, we trade pullbacks to the teeth (8 SMA).
# In ranging markets (all lines intertwined), we fade extremes at 2x ATR from the mid (teeth).
# Volume confirms participation. Weekly EMA50 filters for higher-timeframe trend alignment.
# Designed for low frequency: entries only on strong signals with volume and trend alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Alligator and price
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 6h: SMAs of median price (HL/2)
    median_6h = (high_6h + low_6h) / 2
    lips = pd.Series(median_6h).rolling(window=5, min_periods=5).mean().values      # 5-period
    teeth = pd.Series(median_6h).rolling(window=8, min_periods=8).mean().values      # 8-period
    jaw = pd.Series(median_6h).rolling(window=13, min_periods=13).mean().values      # 13-period
    
    # Alligator alignment: jaw > teeth > lips = bullish, jaw < teeth < lips = bearish
    # Range condition: max-min of the three lines < threshold (indicating intertwined)
    alligator_range = np.maximum.reduce([jaw, teeth, lips]) - np.minimum.reduce([jaw, teeth, lips])
    
    # Volatility measure: ATR(14) on 6h for dynamic thresholds
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.concatenate([[np.nan], close_6h[:-1]]))
    tr3 = np.abs(low_6h - np.concatenate([[np.nan], close_6h[:-1]]))
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    alligator_range_aligned = align_htf_to_ltf(prices, df_6h, alligator_range)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(alligator_range_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Determine market regime
        is_trending = alligator_range_aligned[i] > (1.5 * atr_aligned[i])  # Lines separated
        is_bullish = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        is_bearish = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        # Volume confirmation
        vol_surge = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Trend filter: weekly EMA50
        above_weekly = close[i] > ema50_1w_aligned[i]
        below_weekly = close[i] < ema50_1w_aligned[i]
        
        if is_trending:
            # Trending market: trade pullbacks to teeth (8 SMA)
            # Long: bullish trend + price at or below teeth + volume + weekly alignment
            if is_bullish and low[i] <= teeth_aligned[i] and vol_surge and above_weekly and position <= 0:
                position = 1
                signals[i] = base_size
            # Short: bearish trend + price at or above teeth + volume + weekly alignment
            elif is_bearish and high[i] >= teeth_aligned[i] and vol_surge and below_weekly and position >= 0:
                position = -1
                signals[i] = -base_size
        else:
            # Ranging market: fade extremes at 2x ATR from teeth (midline)
            upper_bound = teeth_aligned[i] + (2.0 * atr_aligned[i])
            lower_bound = teeth_aligned[i] - (2.0 * atr_aligned[i])
            
            # Long: price at lower bound + volume + weekly alignment (contrarian)
            if low[i] <= lower_bound and vol_surge and below_weekly and position <= 0:
                position = 1
                signals[i] = base_size
            # Short: price at upper bound + volume + weekly alignment (contrarian)
            elif high[i] >= upper_bound and vol_surge and above_weekly and position >= 0:
                position = -1
                signals[i] = -base_size
        
        # Exit conditions
        if position == 1:
            # Exit long: price crosses back above teeth (in trend) or above upper bound (in range)
            if (is_trending and close[i] > teeth_aligned[i]) or \
               (not is_trending and close[i] > upper_bound):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: price crosses back below teeth (in trend) or below lower bound (in range)
            if (is_trending and close[i] < teeth_aligned[i]) or \
               (not is_trending and close[i] < lower_bound):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_1dVolume_1wEMA_Filter"
timeframe = "6h"
leverage = 1.0