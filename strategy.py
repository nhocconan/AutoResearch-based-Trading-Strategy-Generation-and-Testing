#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX (12,20) zero-line crossover with 1w EMA34 trend filter and volume confirmation (>1.5x 20-period average)
# TRIX is a triple-smoothed EMA momentum oscillator that filters noise and identifies trend changes
# Zero-line crossovers provide clean entry signals with low whipsaw
# 1w EMA34 ensures alignment with weekly trend to avoid counter-trend trades
# Volume confirmation filters weak signals; discrete sizing (0.25) minimizes fee churn
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe

name = "12h_TRIX_1wEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate TRIX (12,20) on 12h timeframe
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change over 1 period
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change(1).values * 100  # Convert to percentage
    
    # Calculate 20-period average volume for confirmation (on 12h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 12*3, 20)  # 1w EMA34, TRIX warmup (3*12 for triple EMA), volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_trix = trix[i]
        curr_trix_prev = trix[i-1] if i > 0 else 0
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR price closes below 1w EMA34
            if curr_trix <= 0 and curr_trix_prev > 0 or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR price closes above 1w EMA34
            if curr_trix >= 0 and curr_trix_prev < 0 or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: TRIX crosses above zero (bullish momentum) + price above 1w EMA34 + volume confirmation
            if (curr_trix > 0 and curr_trix_prev <= 0 and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero (bearish momentum) + price below 1w EMA34 + volume confirmation
            elif (curr_trix < 0 and curr_trix_prev >= 0 and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals