#!/usr/bin/env python3
"""
4h_TRIX_15_Signal_Line_Cross_1dTrend_VolumeSpike
Hypothesis: TRIX (15) crossing its signal line (9) captures momentum shifts. 
Trades only when aligned with 1d EMA50 trend and volume > 1.5x 20-period average. 
Uses fixed position size 0.25 to limit trades and control drawdown. 
Designed for low trade frequency (~25-40/year) to avoid fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15) and signal line (9) on close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())  # percentage change
    
    # Signal line: EMA of TRIX, period 9
    signal_line = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    
    # TRIX histogram = TRIX - signal line (for crossover detection)
    trix_hist = trix - signal_line
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all indicators to primary timeframe (4h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    trix_hist_aligned = align_htf_to_ltf(prices, pd.DataFrame({'trix_hist': trix_hist}), trix_hist.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: TRIX needs 15*3=45, signal line needs 9, volume avg needs 20
    start_idx = max(45, 9, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_hist_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        trix_hist_val = trix_hist_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs 1d EMA50
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            # Long when TRIX histogram crosses above zero (bullish momentum) in uptrend
            if uptrend and vol_conf and trix_hist_val > 0 and trix_hist_aligned[i-1] <= 0:
                signals[i] = size
                position = 1
            # Short when TRIX histogram crosses below zero (bearish momentum) in downtrend
            elif downtrend and vol_conf and trix_hist_val < 0 and trix_hist_aligned[i-1] >= 0:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit when TRIX histogram crosses below zero (momentum fade)
            if trix_hist_val < 0 and trix_hist_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit when TRIX histogram crosses above zero (momentum fade)
            if trix_hist_val > 0 and trix_hist_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_TRIX_15_Signal_Line_Cross_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0