#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend
Hypothesis: Camarilla pivot levels on 1d (R1/S1) act as strong support/resistance. 
Price breaking above R1 or below S1 with volume confirmation and weekly trend alignment
captures momentum moves. Weekly trend filter (EMA34) avoids counter-trend trades.
Designed for 1d timeframe to reduce trade frequency and fee drag.
Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
Works in both bull and bear markets by following the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Camarilla pivot levels (based on previous day)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous day's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    camarilla_range = prev_high - prev_low
    R1 = prev_close + 1.1 * camarilla_range / 12
    S1 = prev_close - 1.1 * camarilla_range / 12
    
    # Volume filter: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA and to have previous day data
    
    for i in range(start_idx, n):
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = R1[i]
        s1_level = S1[i]
        vol_ok = volume_filter[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend (weekly)
            if price > r1_level and vol_ok and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend (weekly)
            elif price < s1_level and vol_ok and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to previous day's close or trend reverses
            if price < prev_close[i] or price < weekly_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to previous day's close or trend reverses
            if price > prev_close[i] or price > weekly_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_R1_S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0