#!/usr/bin/env python3
"""
4h_12h_RSI_Trend_Filter_v1
Hypothesis: Use RSI(2) on 4h for mean-reversion entries in the direction of 12h EMA(50) trend.
Long when RSI(2) < 10 and price > 12h EMA(50); short when RSI(2) > 90 and price < 12h EMA(50).
Add volume confirmation: current volume > 1.3x 20-period average.
Exit when RSI(2) crosses back above 50 (long) or below 50 (short).
Designed for low trade frequency (target: 20-50 per year) to minimize fee drift.
Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_RSI_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # RSI(2) on 4h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral if undefined
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_entry = rsi[i] < 10 and close[i] > ema_50_aligned[i] and vol_ratio[i] > 1.3
        short_entry = rsi[i] > 90 and close[i] < ema_50_aligned[i] and vol_ratio[i] > 1.3
        
        # Exit conditions: RSI crosses back through 50
        long_exit = rsi[i] > 50
        short_exit = rsi[i] < 50
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals