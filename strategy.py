#!/usr/bin/env python3
"""
Hypothesis: 12-hour strategy using 1-day Williams %R for mean reversion in ranging markets.
Long when Williams %R crosses above -80 from below with price above 200-period EMA.
Short when Williams %R crosses below -20 from above with price below 200-period EMA.
Exit when Williams %R returns to -50 level.
Williams %R identifies overbought/oversold conditions; EMA filter ensures trend alignment.
Designed for low turnover: ~15-25 trades/year per symbol to minimize fee drag.
Works in ranging markets via mean reversion and avoids strong trends via EMA filter.
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
    
    # Load 1-day data once for Williams %R and EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = np.full_like(close_1d, np.nan)
    lowest_low = np.full_like(close_1d, np.nan)
    
    for i in range(lookback - 1, len(close_1d)):
        highest_high[i] = np.max(high_1d[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low_1d[i - lookback + 1:i + 1])
    
    williams_r = np.full_like(close_1d, np.nan)
    for i in range(lookback - 1, len(close_1d)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Calculate 200-period EMA for trend filter
    ema_200 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200[199] = np.mean(close_1d[:200])
        multiplier = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_200[i] = (close_1d[i] - ema_200[i - 1]) * multiplier + ema_200[i - 1]
    
    # Align Williams %R and EMA to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_aligned[i]):
            continue
        
        wr = williams_r_aligned[i]
        ema = ema_200_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND price above EMA200
            if i > 0 and not np.isnan(williams_r_aligned[i-1]):
                prev_wr = williams_r_aligned[i-1]
                if (wr > -80 and prev_wr <= -80 and close[i] > ema):
                    position = 1
                    signals[i] = position_size
            # Short: Williams %R crosses below -20 from above AND price below EMA200
            elif i > 0 and not np.isnan(williams_r_aligned[i-1]):
                prev_wr = williams_r_aligned[i-1]
                if (wr < -20 and prev_wr >= -20 and close[i] < ema):
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit: Williams %R returns to -50 level (mean reversion complete)
            if wr >= -50:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Williams %R returns to -50 level (mean reversion complete)
            if wr <= -50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_1d_WilliamsR_EMA200_MeanReversion"
timeframe = "12h"
leverage = 1.0