#!/usr/bin/env python3
# 6h_1d_camarilla_pivot_reversion_v1
# Hypothesis: 6h strategy using daily Camarilla pivot levels for mean reversion in ranging markets.
# Long: Price touches or breaks below daily S3 level with RSI(14) < 30 and volume > 1.2x average.
# Short: Price touches or breaks above daily R3 level with RSI(14) > 70 and volume > 1.2x average.
# Exit: Price returns to daily pivot point (PP) or opposite S3/R3 level is touched.
# Uses daily Camarilla S3/R3 as extreme reversal zones, 6h for execution with RSI and volume filters.
# Target: 50-150 total trades over 4 years (12-37/year) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) for momentum confirmation
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral RSI when insufficient data
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (focus on S3/R3 for reversals)
    r3 = close_1d + range_1d * 1.1 / 4.0
    s3 = close_1d - range_1d * 1.1 / 4.0
    r4 = close_1d + range_1d * 1.1 / 2.0  # For stronger breakouts
    s4 = close_1d - range_1d * 1.1 / 2.0
    
    # Align HTF levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirmed = volume[i] > 1.2 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to daily pivot or breaks below S4 (stronger down)
            if close[i] >= pivot_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to daily pivot or breaks above R4 (stronger up)
            if close[i] <= pivot_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for reversal at extreme Camarilla levels with volume and RSI confirmation
            bullish_reversal = (close[i] <= s3_aligned[i]) and (rsi[i] < 30) and volume_confirmed
            bearish_reversal = (close[i] >= r3_aligned[i]) and (rsi[i] > 70) and volume_confirmed
            
            if bullish_reversal:
                position = 1
                signals[i] = 0.25
            elif bearish_reversal:
                position = -1
                signals[i] = -0.25
    
    return signals