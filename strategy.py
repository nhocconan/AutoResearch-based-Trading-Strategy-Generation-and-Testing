#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Williams %R with volume confirmation.
# Williams %R > -20 indicates overbought, < -80 indicates oversold.
# Long when Williams %R crosses above -80 from below with volume > 1.5x 20-period EMA.
# Short when Williams %R crosses below -20 from above with volume confirmation.
# Exit when Williams %R crosses back through -50 (centerline).
# Designed for low trade frequency (15-25/year) to avoid fee drag. Works in both trending and ranging markets.

name = "6h_1dWilliamsR_Volume"
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    
    # For Williams %R, we need the highest high and lowest low over the lookback period
    # Using rolling window approach
    williams_r = np.full_like(close_1d, np.nan)
    
    for i in range(13, len(close_1d)):  # Start from index 13 for 14-period
        period_high = np.max(high_1d[i-13:i+1])
        period_low = np.min(low_1d[i-13:i+1])
        if period_high != period_low:
            williams_r[i] = -100 * (period_high - close_1d[i]) / (period_high - period_low)
        else:
            williams_r[i] = -50  # Avoid division by zero
    
    # Williams %R signals: > -20 overbought, < -80 oversold
    # Long when WR crosses above -80 from below
    # Short when WR crosses below -20 from above
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: 6h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for Williams %R
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(williams_r_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1] if i > 0 else -50
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from below with volume confirmation
            if (wr > -80 and wr_prev <= -80 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from above with volume confirmation
            elif (wr < -20 and wr_prev >= -20 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (centerline)
            if wr > -50 and wr_prev <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (centerline)
            if wr < -50 and wr_prev >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals