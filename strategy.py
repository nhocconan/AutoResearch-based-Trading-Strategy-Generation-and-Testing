#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = np.concatenate([[np.nan], high[:-1]])
    low_prev = np.concatenate([[np.nan], low[:-1]])
    close_prev = np.concatenate([[np.nan], close[:-1]])
    
    # Camarilla formulas for R1 and S1
    R1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    S1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # 4h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and enough data
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > R1[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and 1d downtrend
            elif close[i] < S1[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < S1[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > R1[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA trend filter and volume confirmation
# - Camarilla R1/S1 levels act as intraday support/resistance derived from prior day's range
# - Breakouts above R1 or below S1 with volume capture momentum after range periods
# - 1d EMA(34) ensures alignment with higher timeframe trend (works in bull/bear markets)
# - Volume spike (1.8x average) confirms institutional participation
# - Position size 0.25 targets 20-40 trades/year, avoiding fee drag
# - Exit when price returns to opposite S1/R1 level or volume wanes provides logical stop/reversal