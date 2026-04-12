#!/usr/bin/env python3
"""
1d_1w_Trend_Following_with_Volume_Confirmation_v1
Hypothesis: Use 1-week EMA trend filter with daily EMA crossovers and volume confirmation.
Go long when daily EMA(9) crosses above EMA(21) AND price > weekly EMA(50) AND volume > 1.5x average.
Go short when daily EMA(9) crosses below EMA(21) AND price < weekly EMA(50) AND volume > 1.5x average.
Exit when opposite crossover occurs. Designed for low trade frequency (<20 trades/year) to minimize fee drag.
Works in bull via trend following, in bear via shorting counter-trend bounces against weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Trend_Following_with_Volume_Confirmation_v1"
timeframe = "1d"
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
    
    # Daily EMA indicators (need enough data)
    close_series = pd.Series(close)
    ema9 = close_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Daily EMA crossover signals
        ema9_cross_above = ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1]
        ema9_cross_below = ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1]
        
        # Trend filter: price relative to weekly EMA(50)
        price_above_weekly = close[i] > weekly_ema50_aligned[i]
        price_below_weekly = close[i] < weekly_ema50_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_ratio[i] > 1.5
        
        # Entry conditions
        long_entry = ema9_cross_above and price_above_weekly and volume_confirm
        short_entry = ema9_cross_below and price_below_weekly and volume_confirm
        
        # Exit conditions: opposite crossover
        long_exit = ema9_cross_below
        short_exit = ema9_cross_above
        
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