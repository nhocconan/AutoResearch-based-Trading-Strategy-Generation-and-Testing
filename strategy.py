#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA50 trend filter with volume confirmation.
Long when Williams %R(14) crosses above -80 (oversold bounce) AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when Williams %R(14) crosses below -20 (overbought rejection) AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when Williams %R crosses -50 (mean reversion) or opposite signal occurs.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-35 trades/year per symbol.
Williams %R captures short-term reversals in ranging markets, while 1d EMA50 filters for higher timeframe trend alignment.
Volume confirmation ensures only institutional-grade moves are traded. Works in both bull (buy dips) and bear (sell rallies) regimes.
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
    
    # Load 1d data for EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14, 50)  # Ensure warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) AND close > 1d EMA50 AND volume spike
            if (wr > -80 and wr_prev <= -80 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) AND close < 1d EMA50 AND volume spike
            elif (wr < -20 and wr_prev >= -20 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses -50 (mean reversion)
            if position == 1 and wr < -50 and wr_prev >= -50:
                exit_signal = True
            elif position == -1 and wr > -50 and wr_prev <= -50:
                exit_signal = True
            # Alternative exit: opposite signal
            elif position == 1 and (wr < -20 and wr_prev >= -20):
                exit_signal = True
            elif position == -1 and (wr > -80 and wr_prev <= -80):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0