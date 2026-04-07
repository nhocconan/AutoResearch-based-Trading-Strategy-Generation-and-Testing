#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: Camarilla pivot levels from 1-day + EMA trend + volume confirmation on 12h.
Long when price closes above Camarilla R3 with volume above average and price above 1d EMA50.
Short when price closes below Camarilla S3 with volume above average and price below 1d EMA50.
Designed for 15-25 trades/year on 12h timeframe with clear pivot-based mean reversion in ranging markets and trend following in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for pivot calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1-day OHLC
    # Using previous day's close, high, low (already available in df_1d)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Camarilla formula: range = high - low
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 2
    s3 = prev_close - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1-day EMA50 trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price relative to Camarilla levels
        above_r3 = close[i] > r3_aligned[i]
        below_s3 = close[i] < s3_aligned[i]
        
        # 1d trend filter
        above_ema50 = close[i] > ema50_1d_aligned[i]
        below_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 or trend turns bearish
            if below_s3 or below_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or trend turns bullish
            if above_r3 or above_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price closes above R3 with volume confirmation and bullish trend
            if above_r3 and vol_confirmed and above_ema50:
                position = 1
                signals[i] = 0.25
            # Short: price closes below S3 with volume confirmation and bearish trend
            elif below_s3 and vol_confirmed and below_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals