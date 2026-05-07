#!/usr/bin/env python3
name = "12h_Powell_Trap_Midnight"
timeframe = "12h"
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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily EMA for structure
    ema_50_d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Hour-based session filter (UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_50_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_midnight_window = (hour >= 21) or (hour <= 3)  # 9PM-3AM UTC
        
        if position == 0:
            # Long: Powell Trap - bullish reversal at weekly trend support during low liquidity
            if (close[i] > ema_50_d[i] and 
                ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] and  # Weekly uptrend
                in_midnight_window):
                signals[i] = 0.25
                position = 1
            # Short: Powell Trap - bearish rejection at weekly trend resistance during low liquidity
            elif (close[i] < ema_50_d[i] and 
                  ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] and  # Weekly downtrend
                  in_midnight_window):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: weekly trend breaks or price returns to EMA50
            if (ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] or 
                close[i] < ema_50_d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: weekly trend breaks or price returns to EMA50
            if (ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] or 
                close[i] > ema_50_d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Powell Trap exploits institutional stop hunts during low-liquidity midnight sessions
# - During 21:00-03:00 UTC (Asian session overlap), liquidity thins and stops get hunted
# - In weekly uptrend: price dips to EMA50 then reverses up (long trap)
# - In weekly downtrend: price spikes to EMA50 then reverses down (short trap)
# - Weekly EMA20 filter ensures we only trade with the higher timeframe trend
# - EMA50 acts as dynamic support/resistance where stops accumulate
# - Midnight session filter targets periods of lowest liquidity for maximum effect
# - Position size 0.25 limits risk while allowing meaningful moves
# - Designed for 12h timeframe to capture these infrequent but high-probability events
# - Works in both bull (buy the dip) and bear (sell the rally) markets
# - Target: 15-30 trades/year to avoid fee drag while capturing asymmetric moves