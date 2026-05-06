#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and ATR-based stop
# Elder Ray = Bull Power (High - EMA13), Bear Power (Low - EMA13)
# Long when Bull Power > 0 AND Bear Power rising AND price > 1w EMA34 (uptrend)
# Short when Bear Power < 0 AND Bull Power falling AND price < 1w EMA34 (downtrend)
# Exit when power reverses sign or price crosses 1w EMA34 opposite
# Uses 13-period EMA for Elder Ray calculation (standard)
# Discrete sizing 0.25 to balance return and drawdown; target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear: trend filter avoids counter-trend trades, Elder Ray measures underlying pressure

name = "6h_ElderRay_1wEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 34 or len(df_1d) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w EMA34 trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d EMA13 for Elder Ray (standard period)
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align HTF indicators to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND Bear Power rising (less selling pressure) AND uptrend
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] > bear_power_aligned[i-1] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) AND Bull Power falling (less buying pressure) AND downtrend
            elif bear_power_aligned[i] < 0 and bull_power_aligned[i] < bull_power_aligned[i-1] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR price crosses below 1w EMA34 (trend change)
            if bull_power_aligned[i] <= 0 or close[i] <= ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR price crosses above 1w EMA34 (trend change)
            if bear_power_aligned[i] >= 0 or close[i] >= ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals