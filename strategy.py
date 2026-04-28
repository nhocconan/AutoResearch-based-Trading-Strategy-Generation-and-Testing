#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard Camarilla calculation
    range_1d = high_1d - low_1d
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # First day uses its own close
    
    # Camarilla levels
    r3 = close_prev + 1.1 * range_1d / 2
    s3 = close_prev - 1.1 * range_1d / 2
    r4 = close_prev + 1.1 * range_1d
    s4 = close_prev - 1.1 * range_1d
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # Choppiness regime filter (daily)
    # CHOP > 61.8 = range, CHOP < 38.2 = trending
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high_1d).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=atr_period, min_periods=atr_period).min().values
    
    chop = 100 * np.log10(atr.sum() / (highest_high - lowest_low)) / np.log10(atr_period)
    chop = pd.Series(chop).rolling(window=atr_period, min_periods=atr_period).mean().values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Regime: trending (CHOP < 38.2) or range (CHOP > 61.8)
    trending_regime = chop_aligned < 38.2
    range_regime = chop_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Entry conditions based on regime
        # In trending regime: breakout trades
        # In range regime: mean reversion at S3/R3
        
        if trending_regime[i]:
            # Trending: breakout trades
            long_breakout = close[i] > r3_aligned[i]
            short_breakout = close[i] < s3_aligned[i]
            
            long_entry = long_breakout and trend_up and volume_filter[i]
            short_entry = short_breakout and trend_down and volume_filter[i]
            
            # Exit on opposite level
            long_exit = close[i] < s3_aligned[i] and position == 1
            short_exit = close[i] > r3_aligned[i] and position == -1
            
        else:  # range regime
            # Range: mean reversion at extremes
            long_entry = (close[i] < s3_aligned[i]) and trend_up and volume_filter[i]
            short_entry = (close[i] > r3_aligned[i]) and trend_down and volume_filter[i]
            
            # Exit at middle (mean reversion target)
            long_exit = close[i] > (r3_aligned[i] + s3_aligned[i]) / 2 and position == 1
            short_exit = close[i] < (r3_aligned[i] + s3_aligned[i]) / 2 and position == -1
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_RegimeAdaptive_VolumeFilter"
timeframe = "4h"
leverage = 1.0