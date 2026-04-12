#!/usr/bin/env python3
"""
12h_1d_Camarilla_Reversal_v1
Hypothesis: Price reversals at daily Camarilla pivot levels (S3/S4 for longs, R3/R4 for shorts)
with 12h momentum confirmation work across market regimes. Uses 12h timeframe to reduce
frequency and avoid fee drain. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels (S1,S2,S3,S4 and R1,R2,R3,R4)
    range_ = prev_high - prev_low
    # S1 = C - (H-L)*1.08/12, S2 = C - (H-L)*1.12/6, S3 = C - (H-L)*1.12/4, S4 = C - (H-L)*1.12/2
    # R1 = C + (H-L)*1.08/12, R2 = C + (H-L)*1.12/6, R3 = C + (H-L)*1.12/4, R4 = C + (H-L)*1.12/2
    camarilla_s3 = prev_close - range_ * 1.12 / 4
    camarilla_s4 = prev_close - range_ * 1.12 / 2
    camarilla_r3 = prev_close + range_ * 1.12 / 4
    camarilla_r4 = prev_close + range_ * 1.12 / 2
    
    # Handle invalid ranges (zero range)
    valid_range = range_ > 0
    camarilla_s3 = np.where(valid_range, camarilla_s3, np.nan)
    camarilla_s4 = np.where(valid_range, camarilla_s4, np.nan)
    camarilla_r3 = np.where(valid_range, camarilla_r3, np.nan)
    camarilla_r4 = np.where(valid_range, camarilla_r4, np.nan)
    
    # 12h momentum: 12-period RSI for confirmation
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    avg_loss = loss.ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align Camarilla levels and RSI to 12h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: price at extreme Camarilla levels with RSI confirmation
        # Long when price touches S3/S4 and RSI < 30 (oversold)
        # Short when price touches R3/R4 and RSI > 70 (overbought)
        long_setup = ((low[i] <= camarilla_s3_aligned[i] or low[i] <= camarilla_s4_aligned[i]) and 
                      rsi_aligned[i] < 30)
        short_setup = ((high[i] >= camarilla_r3_aligned[i] or high[i] >= camarilla_r4_aligned[i]) and 
                       rsi_aligned[i] > 70)
        
        # Exit conditions: return to midpoint or opposite extreme
        camarilla_midpoint = (camarilla_s4_aligned[i] + camarilla_r4_aligned[i]) / 2
        long_exit = close[i] >= camarilla_midpoint
        short_exit = close[i] <= camarilla_midpoint
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals