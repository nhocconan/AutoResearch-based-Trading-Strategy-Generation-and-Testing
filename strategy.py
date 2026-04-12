#!/usr/bin/env python3
# 12h_1d_camarilla_breakout_volume_regime
# Hypothesis: 12-hour Camarilla breakout with volume confirmation and volatility filter.
# Uses daily Camarilla levels from prior day, requires volume > 1.5x 20-period average,
# and ATR > 0 to avoid low-volatility false breaks. Designed for 12h timeframe
# to target 12-37 trades/year (50-150 total over 4 years) and minimize fee drag.
# Works in bull/bear by filtering breakouts with volume and volatility.

name = "12h_1d_camarilla_breakout_volume_regime"
timeframe = "12h"
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
    
    # Get daily data for Camarilla and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first value where roll creates NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate range
    range_ = prev_high - prev_low
    
    # Camarilla levels (based on previous day)
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # ATR for volatility filter (14-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First TR is invalid due to roll
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Camarilla levels and ATR to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above R4 with volume and volatility filter
        if (close[i] > r4_aligned[i] and vol_confirm[i] and 
            atr_aligned[i] > 0 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume and volatility filter
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and 
              atr_aligned[i] > 0 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals