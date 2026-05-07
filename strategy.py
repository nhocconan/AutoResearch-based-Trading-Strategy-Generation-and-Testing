#!/usr/bin/env python3
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily trend: EMA(34) on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Liquidity sweep detection: look for false breaks of daily high/low
    # Daily high/low from previous day
    daily_high_prev = df_1d['high'].shift(1).values
    daily_low_prev = df_1d['low'].shift(1).values
    daily_high_prev_aligned = align_htf_to_ltf(prices, df_1d, daily_high_prev)
    daily_low_prev_aligned = align_htf_to_ltf(prices, df_1d, daily_low_prev)
    
    # Volume filter: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(daily_high_prev_aligned[i]) or 
            np.isnan(daily_low_prev_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: false break below daily low then reversal with volume
            false_break_low = low[i] < daily_low_prev_aligned[i] and close[i] > daily_low_prev_aligned[i]
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if false_break_low and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: false break above daily high then reversal with volume
            elif high[i] > daily_high_prev_aligned[i] and close[i] < daily_high_prev_aligned[i]:
                if vol_condition and not uptrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: price reaches daily high or momentum fails
            if close[i] >= daily_high_prev_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches daily low or momentum fails
            if close[i] <= daily_low_prev_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Liquidity sweep reversal on 6h timeframe with daily trend filter
# - Looks for false breaks of previous day's high/low (liquidity sweeps)
# - Enters on reversal back inside the prior day's range with volume confirmation
# - Uses daily EMA(34) for trend filter: only long in uptrend, short in downtrend
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Exit when price reaches opposite daily extreme or trend changes
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Liquidity sweeps are common in crypto as stop hunts before reversals