#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_v1
Hypothesis: Daily Camarilla R1/S1 breakouts with weekly EMA34 trend filter. 
In bull regime (price > weekly EMA34), take longs on R1 breakouts; in bear regime (price < weekly EMA34), take shorts on S1 breakdowns. 
Designed for low frequency (target: 30-100 trades over 4 years) to minimize fee drag and work in both bull and bear markets via regime-adaptive directionality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Weekly EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if HTF EMA not ready
        if np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need previous bar's OHLC for Camarilla (bar i-1 must exist)
        if i == 0:
            continue
            
        price = prices['close'].iloc[i]
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        prev_close = prices['close'].iloc[i-1]
        ema_34_1w_val = ema_34_1w_aligned[i]
        
        # Calculate Camarilla levels from previous bar
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
        s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
        
        # Trend regime
        is_bull = price > ema_34_1w_val
        is_bear = price < ema_34_1w_val
        
        if position == 0:
            if is_bull:
                # Bull regime: favor longs on R1 breakouts
                if price > r1:
                    signals[i] = 0.25
                    position = 1
            else:  # bear regime
                # Bear regime: favor shorts on S1 breakdowns
                if price < s1:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions: reverse signal or price fails to hold breakout/breakdown
            if position == 1:
                if price < s1 or price < ema_34_1w_val:  # break below S1 or trend fails
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > r1 or price > ema_34_1w_val:  # break above R1 or trend fails
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_v1"
timeframe = "1d"
leverage = 1.0