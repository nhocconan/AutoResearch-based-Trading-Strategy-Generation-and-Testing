#!/usr/bin/env python3
"""
1d Weekly Range Breakout with Volume and Trend Filter
Long: Price breaks above prior week high + volume > 1.5x 1d volume MA + price > 1d EMA50
Short: Price breaks below prior week low + volume > 1.5x 1d volume MA + price < 1d EMA50
Exit: Opposite break of prior week level
Targets 10-25 trades/year per symbol, designed for both bull and bear markets
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
    
    # Get weekly data for prior week high/low
    df_weekly = get_htf_data(prices, '1w')
    prior_week_high = df_weekly['high'].shift(1)  # Prior week's high
    prior_week_low = df_weekly['low'].shift(1)    # Prior week's low
    ema_50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    prior_week_high_aligned = align_htf_to_ltf(prices, df_weekly, prior_week_high.values)
    prior_week_low_aligned = align_htf_to_ltf(prices, df_weekly, prior_week_low.values)
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Daily volume moving average (20-period for confirmation)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_week_high_aligned[i]) or np.isnan(prior_week_low_aligned[i]) or 
            np.isnan(ema_50_weekly_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        if position == 0:
            # Long: break above prior week high + volume + weekly trend
            if price > prior_week_high_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior week low + volume + weekly trend
            elif price < prior_week_low_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior week low
            if price < prior_week_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior week high
            if price > prior_week_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRange_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0