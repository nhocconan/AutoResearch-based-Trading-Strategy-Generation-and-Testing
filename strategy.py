#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1w EMA50 trend filter and volume spike confirmation.
Long when Williams %R crosses above -80 from below AND close > 1w EMA50 AND volume > 2.0x 20-period average.
Short when Williams %R crosses below -20 from above AND close < 1w EMA50 AND volume > 2.0x 20-period average.
Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts).
Williams %R is a momentum oscillator that identifies overbought/oversold conditions.
In strong trends, it can remain in extreme territory, but reversals from these levels often signal counter-trend moves or pullback endings.
Combined with 1w EMA50 trend filter ensures we trade in the direction of the weekly trend.
Volume confirmation ensures institutional participation. Designed for 6h timeframe to capture medium-term swings.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on primary timeframe (6h)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align HTF indicators to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 14)  # Ensure warmup for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND close > 1w EMA50 AND volume spike
            if (wr > -80 and wr_prev <= -80 and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND close < 1w EMA50 AND volume spike
            elif (wr < -20 and wr_prev >= -20 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses above -20 (for longs) or below -80 (for shorts)
            if position == 1 and wr > -20 and wr_prev <= -20:
                exit_signal = True
            elif position == -1 and wr < -80 and wr_prev >= -80:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0