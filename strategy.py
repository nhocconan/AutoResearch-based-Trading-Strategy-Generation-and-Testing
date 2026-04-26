#!/usr/bin/env python3
"""
6h_RSI2_Extreme_With_1dTrendFilter_v1
Hypothesis: RSI(2) below 10 or above 90 on 6h captures extreme short-term reversals. Filtered by 1d EMA50 trend to trade only in direction of higher timeframe momentum. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend). Uses discrete sizing (0.25) to minimize fee churn. Targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for RSI, EMA
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(2) on 6h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA, 2 for RSI)
    start_idx = max(50, 2)
    
    for i in range(start_idx, n):
        rsi_val = rsi_values[i]
        ema_val = ema_50_1d_aligned[i]
        close_val = close[i]
        
        # Skip if any data not ready
        if np.isnan(ema_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long: RSI(2) < 10 (extreme oversold) and price above 1d EMA50 (uptrend filter)
        long_condition = (rsi_val < 10) and (close_val > ema_val)
        # Short: RSI(2) > 90 (extreme overbought) and price below 1d EMA50 (downtrend filter)
        short_condition = (rsi_val > 90) and (close_val < ema_val)
        
        # Exit: RSI returns to neutral zone (40-60) or opposite extreme
        long_exit = (position == 1 and (rsi_val > 40))
        short_exit = (position == -1 and (rsi_val < 60))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_RSI2_Extreme_With_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0