#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot S1/R1 breakout with volume confirmation and 1d trend filter.
Trades on breakouts of daily pivot support/resistance levels (S1/R1) with volume confirmation
and trend alignment from 1d EMA50. Uses 4h timeframe to keep trade frequency low (target 20-40/year).
Works in bull markets via trend-following breakouts and in bear via mean-reversion at pivot levels.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for price structure
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h OHLC for Camarilla pivot calculation (using previous 4h bar)
    # We need the previous completed 4h bar's OHLC to calculate today's pivots
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    # Set first value to NaN since there's no previous bar
    prev_close_4h[0] = np.nan
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    # Camarilla pivot levels for the current 4h bar based on previous 4h bar
    # R4 = Close + (High-Low) * 1.1/2
    # R3 = Close + (High-Low) * 1.1/4
    # R2 = Close + (High-Low) * 1.1/6
    # R1 = Close + (High-Low) * 1.1/12
    # S1 = Close - (High-Low) * 1.1/12
    # S2 = Close - (High-Low) * 1.1/6
    # S3 = Close - (High-Low) * 1.1/4
    # S4 = Close - (High-Low) * 1.1/2
    hl_range = prev_high_4h - prev_low_4h
    r1 = prev_close_4h + hl_range * 1.1 / 12
    s1 = prev_close_4h - hl_range * 1.1 / 12
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 4h and 1d data to 4h timeframe (we're trading on 4h closes)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above S1 with volume and above 1d EMA50
            if close[i] > s1_aligned[i] and volume_filter[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below R1 with volume and below 1d EMA50
            elif close[i] < r1_aligned[i] and volume_filter[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (mean reversion at support)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (mean reversion at resistance)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S1R1_Breakout_Volume_EMA50"
timeframe = "4h"
leverage = 1.0