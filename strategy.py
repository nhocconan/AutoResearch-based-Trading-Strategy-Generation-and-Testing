#!/usr/bin/env python3
"""
12h_Trix_Volume_WMR_Reversal
Hypothesis: Use TRIX (15-period) momentum reversal with Williams %R oversold/overbought conditions and volume confirmation on 12h timeframe. TRIX filters noise and identifies momentum shifts, while Williams %R identifies overextended conditions. Volume confirmation ensures institutional participation. Works in bull markets by buying oversold dips in uptrend, and in bear markets by selling overbought rallies in downtrend. Targets 15-25 trades/year by requiring TRIX crossover, Williams %R extreme, and volume > 1.5x average.
"""

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
    
    # Get 12h data for TRIX and Williams %R
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate TRIX (15-period)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period percent change
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    trix_raw = ema3.pct_change() * 100  # percentage change
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # Align TRIX and Williams %R to 12h timeframe (no additional delay needed as they are based on current bar)
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix_raw.values)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r.values)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need TRIX and Williams %R warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above zero (bullish momentum), Williams %R oversold (< -80), with volume
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                williams_r_aligned[i] < -80 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero (bearish momentum), Williams %R overbought (> -20), with volume
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  williams_r_aligned[i] > -20 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: TRIX crosses below zero or Williams %R becomes overbought
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above zero or Williams %R becomes oversold
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Trix_Volume_WMR_Reversal"
timeframe = "12h"
leverage = 1.0