#!/usr/bin/env python3
"""
12h_Pivot_R2_S2_Breakout_Volume_Trend
Hypothesis: On 12-hour candles, price breakouts above daily R2 or below daily S2 with volume confirmation and daily EMA trend filter capture institutional momentum. Works in bull/bear by following institutional flow. Target: 15-30 trades/year (60-120 total) to minimize fee drag.
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
    
    # 1-day data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align to 12h timeframe (waits for 1-day bar to close)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 1-day EMA trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_1d_12h[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        r2_val = r2_12h[i]
        s2_val = s2_12h[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_12h[i]
        
        if position == 0:
            # Long: break above R2 with volume in uptrend
            if price > r2_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: break below S2 with volume in downtrend
            elif price < s2_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day)
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                signals[i] = 0.25
                # Exit: price returns to S2 or trend reverses
                if price < s2_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars (1 day)
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                signals[i] = -0.25
                # Exit: price returns to R2 or trend reverses
                if price > r2_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_Pivot_R2_S2_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0