#!/usr/bin/env python3
"""
6h_1d_Relative_Strength_Index_v1
Hypothesis: Use 1-day RSI(14) as a regime filter on 60-minute timeframe. In bull markets (1d RSI > 50), 
look for 6-hour bullish momentum when price crosses above 6h EMA(9) with volume confirmation. 
In bear markets (1d RSI < 50), look for 6-hour bearish momentum when price crosses below 6h EMA(9) 
with volume confirmation. Exit when price crosses back over the 6h EMA(9). 
This strategy adapts to market regime using higher timeframe RSI, reducing false signals 
in choppy environments. Designed for moderate trade frequency (20-40/year) by requiring 
both regime alignment and momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Relative_Strength_Index_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-DAY RSI(14) FOR REGIME FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # === 6H EMA(9) FOR MOMENTUM ===
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align 1D RSI to 6H timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume average (20-period for 6h = ~5 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if indicators not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema9[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Momentum: price cross above/below 6h EMA(9)
        price_cross_above = close[i] > ema9[i] and close[i-1] <= ema9[i-1]
        price_cross_below = close[i] < ema9[i] and close[i-1] >= ema9[i-1]
        
        # Regime filter: 1d RSI > 50 = bull, < 50 = bear
        rsi_bull = rsi_1d_aligned[i] > 50
        rsi_bear = rsi_1d_aligned[i] < 50
        
        # Entry conditions
        long_setup = price_cross_above and vol_confirm and rsi_bull
        short_setup = price_cross_below and vol_confirm and rsi_bear
        
        # Exit conditions: price crosses back over EMA(9)
        exit_long = close[i] < ema9[i] and close[i-1] >= ema9[i-1]
        exit_short = close[i] > ema9[i] and close[i-1] <= ema9[i-1]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals