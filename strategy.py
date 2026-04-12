#!/usr/bin/env python3
"""
6h_1d_camarilla_breakout_volume
Uses Camarilla pivot levels from daily timeframe to identify key support/resistance.
Breakouts above R4 or below S4 with volume confirmation indicate strong momentum.
Fade trades at R3/S3 when price shows rejection signals.
Combines trend following and mean reversion for robust performance in bull/bear markets.
"""

name = "6h_1d_camarilla_breakout_volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using close of previous day as base (standard Camarilla)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have invalid data, handled by checks later
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    # Avoid division by zero
    range_ = np.where(range_ == 0, 1e-10, range_)
    
    # Camarilla levels
    r4 = prev_close + range_ * 1.1 / 2
    r3 = prev_close + range_ * 1.1 / 4
    s3 = prev_close - range_ * 1.1 / 4
    s4 = prev_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Price rejection signals at R3/S3 (wick rejection)
    # Bearish rejection at R3: long upper wick
    upper_wick = high - np.maximum(close, open_) if 'open_' in locals() else high - close
    open_ = prices['open'].values
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    
    # Rejection at R3: close below open and long upper wick
    rejection_r3 = (close < open_) & (upper_wick > 2 * (open_ - close))
    # Rejection at S3: close above open and long lower wick
    rejection_s3 = (close > open_) & (lower_wick > 2 * (close - open_))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R4 with volume
        if close[i] > r4_aligned[i] and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short breakdown: price breaks below S4 with volume
        elif close[i] < s4_aligned[i] and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Fade long at S3 rejection: price shows bullish rejection at support
        elif close[i] <= s3_aligned[i] and rejection_s3[i] and position != 1:
            position = 1
            signals[i] = 0.20
        # Fade short at R3 rejection: price shows bearish rejection at resistance
        elif close[i] >= r3_aligned[i] and rejection_r3[i] and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: return to midpoint or opposite rejection
        elif position == 1:
            # Exit long if price returns to midpoint or shows rejection at R3
            midpoint = (r4_aligned[i] + s4_aligned[i]) / 2
            if close[i] <= midpoint or rejection_r3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short if price returns to midpoint or shows rejection at S3
            midpoint = (r4_aligned[i] + s4_aligned[i]) / 2
            if close[i] >= midpoint or rejection_s3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals