#!/usr/bin/env python3
"""
1d_WeeklyHighLow_Breakout_With_Volume_Filter
Hypothesis: Breakouts above weekly high or below weekly low on daily chart, 
filtered by volume spike and weekly trend. Works in bull markets (buy weekly high breaks) 
and bear markets (sell weekly low breaks). Low trade frequency (~10-25/year) to minimize 
fee drag while capturing significant momentum moves.
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
    
    # Get weekly data for high/low levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly high and low levels
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: >2.0x 20-period average on daily
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high[i]) or 
            np.isnan(weekly_low[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wh = weekly_high[i]
        wl = weekly_low[i]
        ema34 = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above weekly high with volume spike and weekly uptrend
            if price > wh and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume spike and weekly downtrend
            elif price < wl and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price drops back below weekly high OR weekly trend turns down
            if price < wh or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price rises back above weekly low OR weekly trend turns up
            if price > wl or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyHighLow_Breakout_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0