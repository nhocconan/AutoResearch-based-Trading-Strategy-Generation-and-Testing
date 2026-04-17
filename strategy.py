#!/usr/bin/env python3
"""
Hypothesis: On the 12-hour timeframe, price respects weekly high/low levels as key support/resistance.
We combine weekly high/low breakouts with a daily EMA34 trend filter and volume confirmation.
Long when price breaks above prior weekly high with volume > 1.8x average and price above daily EMA34.
Short when price breaks below prior weekly low with volume > 1.8x average and price below daily EMA34.
Exit when price returns to the prior weekly midpoint (mean reversion) or on opposite breakout.
Designed for 12h to capture medium-term trends with ~15-30 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for prior week's high/low
    df_1w = get_htf_data(prices, '1w')
    
    # Prior week high and low (use shift(1) to avoid look-ahead)
    pwhigh = df_1w['high'].shift(1).values
    pwlow = df_1w['low'].shift(1).values
    
    # Prior week midpoint for mean reversion exit
    pwmid = (pwhigh + pwlow) / 2
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    dclose = df_1d['close'].values
    
    # Calculate daily EMA34
    ema_34 = pd.Series(dclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all levels to 12h timeframe
    pwhigh_12h = align_htf_to_ltf(prices, df_1w, pwhigh)
    pwlow_12h = align_htf_to_ltf(prices, df_1w, pwlow)
    pwmid_12h = align_htf_to_ltf(prices, df_1w, pwmid)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pwhigh_12h[i]) or np.isnan(pwlow_12h[i]) or np.isnan(pwmid_12h[i]) or
            np.isnan(ema_34_12h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price breaks above prior weekly high with volume spike and above daily EMA34
            if price > pwhigh_12h[i] and vol > 1.8 * vol_ma and price > ema_34_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior weekly low with volume spike and below daily EMA34
            elif price < pwlow_12h[i] and vol > 1.8 * vol_ma and price < ema_34_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to prior weekly midpoint (mean reversion) OR breaks below prior weekly low
            if price < pwmid_12h[i] or price < pwlow_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to prior weekly midpoint (mean reversion) OR breaks above prior weekly high
            if price > pwmid_12h[i] or price > pwhigh_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyHL_DailyEMA34_Volume"
timeframe = "12h"
leverage = 1.0