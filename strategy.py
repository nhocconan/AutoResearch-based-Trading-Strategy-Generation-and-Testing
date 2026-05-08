#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume spike confirmation.
# Long when price crosses below S3 then reverses above S3 (mean reversion in uptrend).
# Short when price crosses above R3 then reverses below R3 (mean reversion in downtrend).
# Uses 1d EMA34 for trend filter and volume > 1.5x 20-period average for confirmation.
# Target: 15-35 trades/year (60-140 total over 4 years) to avoid fee drag.
# Camarilla levels provide precise intraday support/resistance. Trend filter ensures alignment with higher timeframe direction.
# Volume spike confirms institutional interest in the reversal.

name = "4h_Camarilla_R3S3_Reversal_1dEMA34_Volume"
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
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # True range for Camarilla calculation
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - prev_close)
    tr3 = np.abs(prev_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous day)
    camarilla_h4 = prev_close + (1.1/1.2) * tr
    camarilla_l4 = prev_close - (1.1/1.2) * tr
    camarilla_h3 = prev_close + (1.1/6) * tr
    camarilla_l3 = prev_close - (1.1/6) * tr
    camarilla_h2 = prev_close + (1.1/4) * tr
    camarilla_l2 = prev_close - (1.1/4) * tr
    camarilla_h1 = prev_close + (1.1/2) * tr
    camarilla_l1 = prev_close - (1.1/2) * tr
    
    # Focus on H3/L3 levels for reversals
    camarilla_h3 = camarilla_h3
    camarilla_l3 = camarilla_l3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price crosses above L3 after being below it (reversal from support)
            # AND 1d EMA34 rising (uptrend) AND volume filter
            long_cond = (close[i] > camarilla_l3_aligned[i]) and (close[i-1] <= camarilla_l3_aligned[i-1]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price crosses below H3 after being above it (reversal from resistance)
            # AND 1d EMA34 falling (downtrend) AND volume filter
            short_cond = (close[i] < camarilla_h3_aligned[i]) and (close[i-1] >= camarilla_h3_aligned[i-1]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below L3 (break of support)
            if close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above H3 (break of resistance)
            if close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals