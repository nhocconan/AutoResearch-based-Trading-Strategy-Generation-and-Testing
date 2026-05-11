#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Reversal_Boundary_VolumeFilter
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance in trending markets. 
In strong trends (12h EMA50), price often retraces to R3/S3 before continuing. 
Enter long near S3 in uptrend, short near R3 in downtrend with volume confirmation. 
Use 12h trend filter to avoid counter-trend trades. Target 20-40 trades/year for low friction.
Works in bull/bear by trading with the 12h trend, using Camarilla as dynamic retracement zones.
"""

name = "4h_Camarilla_R3_S3_Reversal_Boundary_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # === Camarilla Pivot Levels from Previous Day ===
    # Use prior day's OHLC to calculate today's Camarilla levels
    # Shift by 1 to avoid look-ahead: use previous day's data
    prev_day_high = pd.Series(high).shift(1)
    prev_day_low = pd.Series(low).shift(1)
    prev_day_close = pd.Series(close).shift(1)
    
    # Typical price for pivot calculation
    typical_price = (prev_day_high + prev_day_low + prev_day_close) / 3
    range_val = prev_day_high - prev_day_low
    
    # Camarilla levels
    R3 = typical_price + range_val * 1.1 / 2
    S3 = typical_price - range_val * 1.1 / 2
    
    # Forward fill to get levels for current day
    R3 = R3.ffill().values
    S3 = S3.ffill().values
    
    # === 12h EMA50 Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers shift and EMA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema50_12h_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price near S3 in uptrend (above 12h EMA50) with volume
            # Entry zone: S3 to S3*1.005 (allow small buffer)
            if (close[i] >= S3[i] and close[i] <= S3[i] * 1.005 and
                close[i] > ema50_12h_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price near R3 in downtrend (below 12h EMA50) with volume
            # Entry zone: R3*0.995 to R3
            elif (close[i] <= R3[i] and close[i] >= R3[i] * 0.995 and
                  close[i] < ema50_12h_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses the opposite Camarilla level
            if position == 1:
                if close[i] >= R3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] <= S3[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals