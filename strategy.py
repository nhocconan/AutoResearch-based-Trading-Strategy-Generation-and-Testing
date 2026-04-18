#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with Volume and 1d Trend Filter
Hypothesis: Camarilla pivot levels act as strong support/resistance levels from institutional
activity. Breaking above R3 or below S3 with volume confirmation indicates institutional
participation. Using 1d EMA50 as trend filter ensures we trade with the dominant daily trend,
which works in both bull and bear markets by filtering counter-trend moves. This combination
provides high-probability entries with controlled frequency (~25-35 trades/year) to minimize
fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate previous day's Camarilla pivot levels
    # Using prior day's high, low, close (available at 4h bar close)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero and NaN
    valid = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close) | 
              (prev_high == prev_low))
    
    # Camarilla formulas
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    R2 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S2 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    PP = (prev_high + prev_low + prev_close) / 3
    
    # Align pivot levels to 4h timeframe (available after daily close)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Volume filter: current volume > 1.8x 24-period volume average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        trend = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: break above R3 with volume, in uptrend (price > EMA50)
            if price > R3_aligned[i] and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume, in downtrend (price < EMA50)
            elif price < S3_aligned[i] and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to pivot point or trend weakens
            if price < PP_aligned[i] or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to pivot point or trend weakens
            if price > PP_aligned[i] or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0