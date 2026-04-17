#!/usr/bin/env python3
"""
1d_WeeklyBreakout_Volume_Trend
Hypothesis: On the daily timeframe, weekly high/low breaks with volume confirmation and weekly trend filter capture sustained moves in both bull and bear markets. 
Weekly high/low acts as institutional support/resistance; volume confirms institutional participation; weekly EMA34 ensures trend alignment.
Designed for 1d to capture multi-day trends with ~10-25 trades per year, avoiding overtrading via strict breakout conditions.
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
    
    # Get weekly data for prior week's high/low and EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    
    # Prior weekly high and low (use shift(1) to avoid look-ahead: use completed week's levels)
    pweekly_high = df_1w['high'].shift(1).values
    pweekly_low = df_1w['low'].shift(1).values
    pweekly_close = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter (use prior week's close)
    ema_34_1w = pd.Series(pweekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all weekly levels to daily timeframe (waits for weekly bar to close)
    pweekly_high_d = align_htf_to_ltf(prices, df_1w, pweekly_high)
    pweekly_low_d = align_htf_to_ltf(prices, df_1w, pweekly_low)
    ema_34_1w_d = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 20-period volume MA on daily
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pweekly_high_d[i]) or np.isnan(pweekly_low_d[i]) or np.isnan(ema_34_1w_d[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior weekly high with volume spike and above weekly EMA34
            if price > pweekly_high_d[i] and vol > 2.0 * vol_ma and price > ema_34_1w_d[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior weekly low with volume spike and below weekly EMA34
            elif price < pweekly_low_d[i] and vol > 2.0 * vol_ma and price < ema_34_1w_d[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior weekly low (trailing stop) or opposite breakout
            if price < pweekly_low_d[i] or price < pweekly_high_d[i] * 0.98:  # 2% buffer from weekly high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior weekly high (trailing stop) or opposite breakout
            if price > pweekly_high_d[i] or price > pweekly_low_d[i] * 1.02:  # 2% buffer from weekly low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBreakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0