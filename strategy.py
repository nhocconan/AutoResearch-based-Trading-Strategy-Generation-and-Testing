#!/usr/bin/env python3
name = "4h_InsideBar_Reversal_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume spike detection
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Inside bar: current range within previous bar
        inside_bar = (high[i] <= high[i-1]) and (low[i] >= low[i-1])
        
        vol_condition = volume[i] > vol_ma_10[i] * 1.5
        
        if position == 0:
            # Long: inside bar reversal up in daily uptrend
            if inside_bar and close[i] > close[i-1] and vol_condition and ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: inside bar reversal down in daily downtrend
            elif inside_bar and close[i] < close[i-1] and vol_condition and ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below inside bar low
            if close[i] < low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above inside bar high
            if close[i] > high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Inside bar reversals on 4h with daily trend filter and volume confirmation
# - Inside bar: consolidation with lower volatility, often precedes breakout
# - Breakout direction determined by close relative to previous close
# - Daily EMA20 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price returns to opposite side of inside bar range
# - Works in both bull (long reversals in uptrend) and bear (short reversals in downtrend)
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Inside bars provide clear structure with defined support/resistance levels
# - Daily trend filter reduces whipsaws vs same-timeframe signals
# - Novel combination: Inside bar reversals (4h) + daily trend + volume spike not recently tried
# - Aims for 80-200 total trades over 4 years (20-50/year) to stay within limits