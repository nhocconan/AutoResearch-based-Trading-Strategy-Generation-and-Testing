#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R1_S1_Breakout
Hypothesis: Weekly Camarilla pivot levels (R1, S1) from the previous week act as key support/resistance.
Breakouts above weekly R1 or below weekly S1 on daily close with volume confirmation and weekly trend filter
capture institutional moves. Works in bull/bear by following smart money. Target: 10-25 trades/year (40-100 total over 4 years).
"""

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
    
    # Weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Weekly Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to daily timeframe (waits for weekly bar to close)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume filter: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Weekly EMA trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_1d = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1w_1d[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        r1_val = r1_1d[i]
        s1_val = s1_1d[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1w_1d[i]
        
        if position == 0:
            # Long: break above R1 with volume in uptrend
            if price > r1_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below S1 with volume in downtrend
            elif price < s1_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 3 days
            if bars_since_entry < 3:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to S1 or trend reverses
                if price < s1_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 3 days
            if bars_since_entry < 3:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to R1 or trend reverses
                if price > r1_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "1d_Weekly_Camarilla_R1_S1_Breakout"
timeframe = "1d"
leverage = 1.0