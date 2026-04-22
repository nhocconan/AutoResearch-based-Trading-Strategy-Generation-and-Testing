#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot reversal with 1-day trend filter and volume confirmation.
Long when price touches S3 level with bullish 1d EMA trend and volume spike.
Short when price touches R3 level with bearish 1d EMA trend and volume spike.
Exit when price moves toward mean (Pivot) or trend weakens.
Camarilla pivots identify key intraday support/resistance; 1d EMA filters for higher-timeframe trend;
volume spike confirms institutional interest at extremes. Designed for low trade frequency by requiring
multiple confluences and only trading at extreme levels (S3/R3). Works in ranging markets via mean
reversion and in trending markets via trend-filtered breaks.
"""

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
    
    # Calculate Camarilla levels for each bar using previous day's OHLC
    # We need daily OHLC, so we'll get 1d data and align it
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla calculation
    # For each 4h bar, we use the previous completed day's OHLC
    prev_day_open = df_1d['open'].values
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    prev_day_close = df_1d['close'].values
    
    # Align daily data to 4h timeframe (each 4h bar gets the PREVIOUS day's OHLC)
    # Since we want the day that completed before the current 4h bar
    prev_day_open_aligned = align_htf_to_ltf(prices, df_1d, prev_day_open)
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Calculate Camarilla levels
    # R4 = Close + (High-Low) * 1.5000
    # R3 = Close + (High-Low) * 1.2500
    # R2 = Close + (High-Low) * 1.1666
    # R1 = Close + (High-Low) * 1.0833
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High-Low) * 1.0833
    # S2 = Close - (High-Low) * 1.1666
    # S3 = Close - (High-Low) * 1.2500
    # S4 = Close - (High-Low) * 1.5000
    
    rng = prev_day_high_aligned - prev_day_low_aligned
    r3 = prev_day_close_aligned + rng * 1.2500
    s3 = prev_day_close_aligned - rng * 1.2500
    pivot = (prev_day_high_aligned + prev_day_low_aligned + prev_day_close_aligned) / 3.0
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(prev_day_close_aligned).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(pivot[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price touches or goes below S3 with bullish 1d trend and volume spike
            if low[i] <= s3[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price touches or goes above R3 with bearish 1d trend and volume spike
            elif high[i] >= r3[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price moves back toward pivot or trend weakens
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reaches/passes pivot or 1d EMA turns down
                if high[i] >= pivot[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reaches/passes pivot or 1d EMA turns up
                if low[i] <= pivot[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_S3R3_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0