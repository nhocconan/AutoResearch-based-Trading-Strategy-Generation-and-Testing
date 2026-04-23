#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R crosses above -80 from oversold AND close > 1d EMA50 AND volume > 1.5x average.
Short when Williams %R crosses below -20 from overbought AND close < 1d EMA50 AND volume > 1.5x average.
Exit when Williams %R crosses -50 (mean reversion) or opposing signal occurs.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Williams %R identifies exhaustion points in both bull and bear markets. 1d EMA50 provides smooth trend filter.
Volume confirmation ensures institutional participation. Designed for 12h timeframe to reduce trade frequency.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R on primary timeframe (12h)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Align HTF indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Ensure warmup for EMA50, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
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
            # Long: Williams %R crosses above -80 from oversold AND close > 1d EMA50 AND volume spike
            if (wr > -80 and wr_prev <= -80 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought AND close < 1d EMA50 AND volume spike
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
            
            # Secondary exit: opposing signal
            elif position == 1 and wr < -20 and wr_prev >= -20 and volume[i] > 1.5 * vol_ma_val:
                exit_signal = True
            elif position == -1 and wr > -80 and wr_prev <= -80 and volume[i] > 1.5 * vol_ma_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0