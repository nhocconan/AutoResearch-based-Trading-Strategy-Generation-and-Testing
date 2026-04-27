#!/usr/bin/env python3
"""
12h_Pivot_DeMark_Sequential_Setup_Trend_Filter_Volume
Hypothesis: Trade 12h timeframe with Tom DeMark Sequential setup (9/13) for trend exhaustion,
filtered by 1d trend (EMA50) and volume confirmation. TD Sequential identifies exhaustion
points in extended trends, providing high-probability reversal entries. Works in both
bull and bear markets by catching trend reversals at overextended levels.
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
    
    # Get 1d data for TD Sequential and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TD Sequential setup phase (count of 9 consecutive closes)
    # Buy setup: 9 consecutive closes > close 4 bars ago
    # Sell setup: 9 consecutive closes < close 4 bars ago
    close_1d = df_1d['close'].values
    
    # Initialize setup counters
    buy_setup = np.zeros(len(close_1d))
    sell_setup = np.zeros(len(close_1d))
    
    buy_count = 0
    sell_count = 0
    
    for i in range(4, len(close_1d)):
        if close_1d[i] > close_1d[i-4]:
            buy_count += 1
            sell_count = 0
        elif close_1d[i] < close_1d[i-4]:
            sell_count += 1
            buy_count = 0
        else:
            buy_count = 0
            sell_count = 0
        
        # Cap at 9 for setup phase
        buy_setup[i] = min(buy_count, 9)
        sell_setup[i] = min(sell_count, 9)
    
    # TD Sequential signals: buy setup = 9, sell setup = 9
    td_buy_signal = (buy_setup == 9).astype(float)
    td_sell_signal = (sell_setup == 9).astype(float)
    
    # Align TD signals to 12h timeframe
    td_buy_aligned = align_htf_to_ltf(prices, df_1d, td_buy_signal)
    td_sell_aligned = align_htf_to_ltf(prices, df_1d, td_sell_signal)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.8 * 24-period average (on 12h data, ~12 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for TD Sequential, EMA, and volume average
    start_idx = max(24, 50)
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data not ready
        if (np.isnan(td_buy_aligned[i]) or np.isnan(td_sell_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        td_buy = td_buy_aligned[i] > 0.5
        td_sell = td_sell_aligned[i] > 0.5
        ema_50_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Require minimum 24 bars since last exit to avoid churn (~12 days on 12h)
            if bars_since_exit >= 24:
                # Long: TD buy setup (9) + above EMA50 (uptrend) + volume confirmation
                if td_buy and close[i] > ema_50_val and vol_conf:
                    signals[i] = size
                    position = 1
                    bars_since_exit = 0
                # Short: TD sell setup (9) + below EMA50 (downtrend) + volume confirmation
                elif td_sell and close[i] < ema_50_val and vol_conf:
                    signals[i] = -size
                    position = -1
                    bars_since_exit = 0
        elif position == 1:
            # Exit long: TD sell setup (9) or price crosses below EMA50
            if td_sell or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TD buy setup (9) or price crosses above EMA50
            if td_buy or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Pivot_DeMark_Sequential_Setup_Trend_Filter_Volume"
timeframe = "12h"
leverage = 1.0