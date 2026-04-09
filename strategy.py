#!/usr/bin/env python3
# 4h_camarilla_pullback_volume_v2
# Hypothesis: 4h strategy using 1d Camarilla pivot levels with pullback entries and volume confirmation.
# Long when price pulls back to S3/S4 with volume > 1.5x 20-period average during uptrend (close > EMA50).
# Short when price pulls back to R3/R4 with volume > 1.5x 20-period average during downtrend (close < EMA50).
# Exit when price reaches opposite Camarilla level (R1 for longs, S1 for shorts) or breaks through pivot.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture mean-reversion within the day's range while avoiding false breakouts.
# Target: 15-30 trades/year (60-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pullback_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    pivot = typical_price
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    s2 = close_1d - (range_1d * 1.1 / 6)
    r3 = close_1d + (range_1d * 1.1 / 4)
    s3 = close_1d - (range_1d * 1.1 / 4)
    r4 = close_1d + (range_1d * 1.1 / 2)
    s4 = close_1d - (range_1d * 1.1 / 2)
    
    # Align all levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(open_[i]) or np.isnan(ema50[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price reaches R1 (take profit) or breaks below pivot (stop)
            if close[i] >= r1_aligned[i] or close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price reaches S1 (take profit) or breaks above pivot (stop)
            if close[i] <= s1_aligned[i] or close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for pullback to support/resistance with volume confirmation and trend filter
            bullish_pullback = (close[i] <= s3_aligned[i] + 0.5 * (s4_aligned[i] - s3_aligned[i])) and \
                              volume_confirmed and (close[i] > ema50[i])
            bearish_pullback = (close[i] >= r3_aligned[i] - 0.5 * (r4_aligned[i] - r3_aligned[i])) and \
                              volume_confirmed and (close[i] < ema50[i])
            
            if bullish_pullback:
                position = 1
                signals[i] = 0.25
            elif bearish_pullback:
                position = -1
                signals[i] = -0.25
    
    return signals