#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Momentum_v1
Hypothesis: In both bull and bear markets, price tends to continue after breaking above/below key daily Camarilla pivot levels (H4/L4) with volume confirmation and trend alignment. Use 4h for entry timing and 1d for pivot levels and trend filter (EMA100). Target 25-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 110:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_s = pd.Series(daily_close)
    daily_high_s = pd.Series(daily_high)
    daily_low_s = pd.Series(daily_low)
    
    # === 1D CAMARILLA PIVOT LEVELS (H4, L4) ===
    # Pivot = (H + L + C) / 3
    pivot = (daily_high + daily_low + daily_close) / 3.0
    # Range = H - L
    range_hl = daily_high - daily_low
    # H4 = Close + Range * 1.1/2
    # L4 = Close - Range * 1.1/2
    h4 = daily_close + range_hl * 1.1 / 2.0
    l4 = daily_close - range_hl * 1.1 / 2.0
    
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    
    # === 1D EMA100 TREND FILTER ===
    ema100 = daily_close_s.ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_4h = align_htf_to_ltf(prices, df_1d, ema100)
    
    # === VOLUME CONFIRMATION (2x 20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(110, n):
        # Skip if any data invalid
        if (np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or 
            np.isnan(ema100_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout above H4 or below L4 with volume and trend alignment
        breakout_long = high[i] > h4_4h[i]
        breakout_short = low[i] < l4_4h[i]
        
        # Trend filter: align with EMA100 direction
        uptrend = close[i] > ema100_4h[i]
        downtrend = close[i] < ema100_4h[i]
        
        # Entry conditions
        long_entry = breakout_long and vol_spike[i] and uptrend
        short_entry = breakout_short and vol_spike[i] and downtrend
        
        # Exit: opposite breakout or loss of trend
        long_exit = breakout_short or not uptrend
        short_exit = breakout_long or not downtrend
        
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