#!/usr/bin/env python3
"""
4h_1d_Weekly_Pullback_To_VWAP_v1
Hypothesis: On 4h timeframe, price often pulls back to weekly VWAP after strong moves.
Enter long when price pulls back to weekly VWAP with bullish volume divergence (volume decreasing on pullback),
short when price rallies to weekly VWAP with bearish volume divergence.
Weekly VWAP acts as dynamic support/resistance; volume divergence indicates weakening momentum.
Designed for low trade frequency (target: 20-60 total over 4 years) to minimize fee drag.
Works in buy the dip scenarios in bull markets and sell the rally in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Weekly_Pullback_To_VWAP_v1"
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
    
    # Weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP using prior week's data
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    weekly_vwap = vwap.iloc[-1]  # Most recent completed week's VWAP
    
    # Align weekly VWAP to 4h timeframe
    weekly_vwap_array = np.full(len(df_1w), weekly_vwap)
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1w, weekly_vwap_array)
    
    # Volume trend: decreasing volume on pullback/rally
    volume_series = pd.Series(volume)
    vol_trend = volume_series.diff(3)  # Change over 3 periods
    vol_decreasing = vol_trend < 0
    
    # Price deviation from weekly VWAP
    price_dev = (close - weekly_vwap_aligned) / weekly_vwap_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if np.isnan(weekly_vwap_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: pullback to VWAP with volume divergence
        pullback_to_vwap = abs(price_dev[i]) < 0.005  # Within 0.5% of VWAP
        long_setup = pullback_to_vwap and price_dev[i] > 0 and vol_decreasing.iloc[i]  # Slightly above VWAP, falling volume
        short_setup = pullback_to_vwap and price_dev[i] < 0 and vol_decreasing.iloc[i]  # Slightly below VWAP, falling volume
        
        # Exit conditions: move away from VWAP
        long_exit = price_dev[i] > 0.015  # 1.5% above VWAP
        short_exit = price_dev[i] < -0.015  # 1.5% below VWAP
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
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