# 4h_ETF_Flow_Pullback_1dTrend_Volume
# Hypothesis: Combines ETF flow proxy (volume imbalance) with 1-day trend and pullback entries.
# Long: Pullback to EMA20 during daily uptrend with buying volume imbalance (buys > sells).
# Short: Pullback to EMA20 during daily downtrend with selling volume imbalance (sells > buys).
# Uses institutional flow signals to capture reversals in trend context.
# Target: 30-50 trades/year to avoid fee drag.

name = "4h_ETF_Flow_Pullback_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA20 on 4h for pullback entries
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume imbalance: buying pressure vs selling pressure
    # taker_buy_volume = volume bought at ask (aggressive buys)
    # volume - taker_buy_volume = volume sold at bid (aggressive sells)
    buy_pressure = taker_buy_volume
    sell_pressure = volume - taker_buy_volume
    volume_imbalance = buy_pressure - sell_pressure  # positive = buying pressure
    
    # Smooth the imbalance to reduce noise
    vol_imb_smooth = pd.Series(volume_imbalance).ewm(span=10, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 10)  # Ensure we have EMA20, EMA50, and smoothed imbalance
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema20[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_imb_smooth[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to EMA20 during daily uptrend with buying pressure
            if (close[i] >= ema20[i] * 0.998 and close[i] <= ema20[i] * 1.002 and  # near EMA20
                close[i] > ema50_1d_aligned[i] and  # daily uptrend
                vol_imb_smooth[i] > 0):  # buying pressure
                signals[i] = 0.25
                position = 1
            # Short: pullback to EMA20 during daily downtrend with selling pressure
            elif (close[i] >= ema20[i] * 0.998 and close[i] <= ema20[i] * 1.002 and  # near EMA20
                  close[i] < ema50_1d_aligned[i] and  # daily downtrend
                  vol_imb_smooth[i] < 0):  # selling pressure
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price moves significantly away from EMA20 or trend changes
            if (close[i] < ema20[i] * 0.97 or  # 3% below EMA20
                close[i] < ema50_1d_aligned[i]):  # trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price moves significantly away from EMA20 or trend changes
            if (close[i] > ema20[i] * 1.03 or  # 3% above EMA20
                close[i] > ema50_1d_aligned[i]):  # trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals