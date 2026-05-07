#/usr/bin/env python3
name = "6h_Liquidity_Sweep_Reversal"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily trend filter: EMA(21) on daily close
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly high/low for liquidity sweep detection
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # ATR(14) on 6h for dynamic thresholds
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 14)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Liquidity sweep conditions
        sweep_up = high[i] > weekly_high_aligned[i] and close[i] < weekly_high_aligned[i]
        sweep_down = low[i] < weekly_low_aligned[i] and close[i] > weekly_low_aligned[i]
        
        if position == 0:
            # Long: liquidity sweep below weekly low with daily uptrend
            if sweep_down and ema_21_1d_aligned[i] > ema_21_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: liquidity sweep above weekly high with daily downtrend
            elif sweep_up and ema_21_1d_aligned[i] < ema_21_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price recovers above weekly low or trend changes
            if close[i] > weekly_low_aligned[i] or ema_21_1d_aligned[i] < ema_21_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price drops below weekly high or trend changes
            if close[i] < weekly_high_aligned[i] or ema_21_1d_aligned[i] > ema_21_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h liquidity sweep reversal with daily trend filter
# - Price often sweeps weekly highs/lows to trigger stops before reversing
# - Long when price sweeps below weekly low then closes back above (bull trap)
# - Short when price sweeps above weekly high then closes back below (bear trap)
# - Daily EMA(21) ensures we trade in direction of higher timeframe trend
# - Works in bull (buy sweeps of lows in uptrend) and bear (sell sweeps of highs in downtrend)
# - Weekly levels provide significant liquidity pools that attract stops
# - Position size 0.25 targets ~20-30 trades/year, avoiding excessive fees
# - ATR would be used for dynamic stop/target but we use weekly levels as natural boundaries