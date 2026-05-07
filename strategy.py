#!/usr/bin/env python3
name = "4h_4H_1D_Trend_With_Filter"
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
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMAs for trend
    ema_9_4h = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21_4h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_9_4h[i]) or 
            np.isnan(ema_21_4h[i]) or np.isnan(vol_ma_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend condition
        trend_up = ema_9_4h[i] > ema_21_4h[i]
        trend_down = ema_9_4h[i] < ema_21_4h[i]
        
        # Volume condition
        vol_condition = volume[i] > vol_ma_20_4h[i] * 1.5
        
        if position == 0:
            # Long: 4h uptrend + daily uptrend + volume spike
            if trend_up and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: 4h downtrend + daily downtrend + volume spike
            elif trend_down and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: trend reversal or volume drop
            if not trend_up or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: trend reversal or volume drop
            if not trend_down or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 4h trend following with daily trend filter and volume confirmation
# - Uses 9/21 EMA crossover on 4h for trend direction
# - Filters with 34 EMA on daily timeframe (aligned) to ensure higher timeframe trend alignment
# - Requires volume spike (1.5x 20-period average) to confirm momentum
# - Works in both bull (long in uptrends) and bear (short in downtrends) markets
# - Position size 0.30 balances risk and return
# - Exit when trend reverses or volume confirmation fails
# - Designed for ~20-40 trades/year to stay within limits and minimize fee drag
# - Daily trend filter reduces whipsaws vs pure 4h signals
# - Volume confirmation reduces false breakouts and improves signal quality
# - Simple, robust logic with clear entry/exit conditions