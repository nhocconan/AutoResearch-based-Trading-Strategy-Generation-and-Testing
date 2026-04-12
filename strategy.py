#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_With_Filter
Hypothesis: Daily Camarilla pivot breakouts with weekly trend filter and volume confirmation.
Go long when price breaks above daily H3 in uptrend, short when breaks below L3 in downtrend.
Weekly trend determined by price position relative to weekly VWAP.
Works in bull (breakouts continuation) and bear (breakdowns continuation).
Target: 50-100 total trades over 4 years (12-25/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_With_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP for trend
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_1w = (typical_price_1w * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_1w_values = vwap_1w.values
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w_values)
    
    # === DAILY CAMARILLA PIVOTS ===
    # Use previous day's OHLC to calculate today's Camarilla levels
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first day uses same day
    
    # Camarilla calculation
    range_val = prev_high - prev_low
    h3 = prev_close + range_val * 1.1 / 6
    l3 = prev_close - range_val * 1.1 / 6
    h4 = prev_close + range_val * 1.1 / 2
    l4 = prev_close - range_val * 1.1 / 2
    
    # === VOLUME FILTER (20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from weekly VWAP
        trend_up = close[i] > vwap_1w_aligned[i]
        trend_down = close[i] < vwap_1w_aligned[i]
        
        # Breakout conditions: price breaks Camarilla H3/L3 + trend + volume
        breakout_up = close[i] > h3[i]
        breakout_down = close[i] < l3[i]
        
        long_signal = breakout_up and trend_up and vol_ratio[i] > 1.5
        short_signal = breakout_down and trend_down and vol_ratio[i] > 1.5
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = (position == 1 and 
                    (breakout_down or not trend_up))
        exit_short = (position == -1 and 
                     (breakout_up or not trend_down))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals