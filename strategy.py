#!/usr/bin/env python3
"""
12h_1W_Camarilla_R1_S1_Breakout_Volume_Trend
Hypothesis: Weekly chart provides primary trend direction for 12h trades. 
Breakouts above weekly R1 or below weekly S1 with volume confirmation on 12h timeframe
capture institutional moves while filtering noise. Weekly timeframe reduces noise
and provides stronger trend context for longer-term moves in both bull and bear markets.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
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
    
    # 1-week data for primary trend and weekly pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly Camarilla
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Weekly Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    weekly_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    weekly_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align weekly levels to 12h timeframe
    weekly_r1_12h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly EMA trend filter (34-period)
    weekly_ema = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_12h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: >1.5x 20-period average on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_r1_12h[i]) or np.isnan(weekly_s1_12h[i]) or
            np.isnan(weekly_ema_12h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = weekly_r1_12h[i]
        s1_val = weekly_s1_12h[i]
        vol_ok = volume_filter[i]
        ema_trend = weekly_ema_12h[i]
        
        if position == 0:
            # Long: break above weekly R1 with volume in uptrend
            if price > r1_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 with volume in downtrend
            elif price < s1_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to weekly S1 or trend reverses
            if price < s1_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to weekly R1 or trend reverses
            if price > r1_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1W_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0