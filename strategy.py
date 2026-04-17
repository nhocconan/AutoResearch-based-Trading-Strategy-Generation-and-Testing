#!/usr/bin/env python3
"""
1D Weekly Range Breakout with Volume and Trend Filter
Long: Price breaks above prior 1-week high + volume > 1.5x 1D volume MA + price > 1W EMA34
Short: Price breaks below prior 1-week low + volume > 1.5x 1D volume MA + price < 1W EMA34
Exit: Opposite break of prior 1-week level
Uses 1W EMA34 for trend alignment to reduce false breakouts in choppy markets
Target: 7-25 trades/year per symbol
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
    
    # Get 1W data for prior weekly high/low and trend filter
    df_1w = get_htf_data(prices, '1w')
    prior_1w_high = df_1w['high'].shift(1)  # Prior week's high
    prior_1w_low = df_1w['low'].shift(1)    # Prior week's low
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    prior_1w_high_aligned = align_htf_to_ltf(prices, df_1w, prior_1w_high.values)
    prior_1w_low_aligned = align_htf_to_ltf(prices, df_1w, prior_1w_low.values)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1D volume moving average (20-period for confirmation)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_1w_high_aligned[i]) or np.isnan(prior_1w_low_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        if position == 0:
            # Long: break above prior 1W high + volume + 1W trend
            if price > prior_1w_high_aligned[i] and vol > 1.5 * vol_ma and price > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below prior 1W low + volume + 1W trend
            elif price < prior_1w_low_aligned[i] and vol > 1.5 * vol_ma and price < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below prior 1W low
            if price < prior_1w_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above prior 1W high
            if price > prior_1w_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyRange_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0