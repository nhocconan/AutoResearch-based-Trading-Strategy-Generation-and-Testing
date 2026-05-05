#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Camarilla levels from 1d provide strong intraday support/resistance
# Volume spike confirms institutional participation
# Choppiness index (CHOP > 61.8) ensures we only trade in ranging markets where mean reversion works
# Long at S3 breakout with volume, short at R3 breakdown with volume
# Exit at opposite Camarilla level (R3 for longs, S3 for shorts) or chop exit
# Works in both bull and bear markets by fading extremes in ranging conditions
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3S3_Breakout_1dVolume_Chop"
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
    
    # Get 1d data ONCE before loop for Camarilla levels, volume MA, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But we use the close of the completed 1d bar to calculate levels for next period
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3 and S3 levels
    rangex = high_1d - low_1d
    r3 = close_1d + 1.1 * rangex
    s3 = close_1d - 1.1 * rangex
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d volume MA for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)  # Require 2x volume spike
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Choppiness Index on 1d (to detect ranging markets)
    # CHOP = 100 * log10(sum(ATR over n) / (n * log(high_n/low_n))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    if len(high_1d) >= 14:
        # Calculate True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum.reduce([tr1, tr2, tr3])
        tr = np.concatenate([[np.nan], tr])  # Align with index
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index
        chop = 100 * np.log10(tr_sum / (max_high - min_low)) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
        chop_ranging = chop > 61.8  # Only trade in ranging markets
        chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging)
    else:
        chop_ranging_aligned = np.ones(n, dtype=bool)  # Default to ranging if not enough data
    
    # 4h volume confirmation (secondary confirmation)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike_4h = volume > (1.5 * vol_ma_20)
    else:
        volume_spike_4h = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(chop_ranging_aligned[i]) if isinstance(chop_ranging_aligned[i], float) else False):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Handle boolean chop_ranging_aligned (could be bool or float)
        chop_ok = chop_ranging_aligned[i]
        if isinstance(chop_ok, np.bool_):
            chop_ok = bool(chop_ok)
        elif isinstance(chop_ok, float) and np.isnan(chop_ok):
            chop_ok = False
        
        if position == 0:
            # Long conditions: Close breaks above S3 with volume spike AND chop ranging
            if (close[i] > s3_aligned[i] and 
                volume_spike_1d_aligned[i] and 
                chop_ok and
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Close breaks below R3 with volume spike AND chop ranging
            elif (close[i] < r3_aligned[i] and 
                  volume_spike_1d_aligned[i] and 
                  chop_ok and
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close crosses below R3 (mean reversion target) OR chop breaks below 38.2 (trend start)
            chop_trending = False
            if isinstance(chop_ranging_aligned[i], float) and not np.isnan(chop_ranging_aligned[i]):
                chop_trending = chop_ranging_aligned[i] < 38.2
            elif isinstance(chop_ranging_aligned[i], np.bool_):
                chop_trending = not bool(chop_ranging_aligned[i])
            
            if close[i] < r3_aligned[i] or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close crosses above S3 (mean reversion target) OR chop breaks below 38.2 (trend start)
            chop_trending = False
            if isinstance(chop_ranging_aligned[i], float) and not np.isnan(chop_ranging_aligned[i]):
                chop_trending = chop_ranging_aligned[i] < 38.2
            elif isinstance(chop_ranging_aligned[i], np.bool_):
                chop_trending = not bool(chop_ranging_aligned[i])
            
            if close[i] > s3_aligned[i] or chop_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals