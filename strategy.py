#!/usr/bin/env python3
"""
4h_12h_camarilla_ema50_volume_v2
Hypothesis: Reuse winning pattern (Camarilla + EMA trend + volume) but tighten entry with 2-bar close confirmation and add ADX(14) filter to avoid chop. Target 15-30 trades/year to avoid fee drag. Works in bull/bear via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and Camarilla
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend direction
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Previous 12h bar's range for Camarilla
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    
    range_12h = prev_high_12h - prev_low_12h
    # Resistance levels
    r3 = prev_close_12h + range_12h * 1.1 / 2
    r4 = prev_close_12h + range_12h * 1.1
    # Support levels
    s3 = prev_close_12h - range_12h * 1.1 / 2
    s4 = prev_close_12h - range_12h * 1.1
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # ADX(14) filter on 4h to avoid chop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: ADX > 25
    trend_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    consecutive_high = np.zeros(n, dtype=int)  # count consecutive closes > R4
    consecutive_low = np.zeros(n, dtype=int)   # count consecutive closes < S4
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(trend_filter[i])):
            signals[i] = 0.0
            continue
        
        # Count consecutive closes above/below levels for confirmation
        if close[i] > r4_aligned[i]:
            consecutive_high[i] = consecutive_high[i-1] + 1 if i > 0 else 1
        else:
            consecutive_high[i] = 0
            
        if close[i] < s4_aligned[i]:
            consecutive_low[i] = consecutive_low[i-1] + 1 if i > 0 else 1
        else:
            consecutive_low[i] = 0
        
        # Long entry: price > EMA50 (uptrend) AND 2 consecutive closes > R4 with volume AND trend
        if (close[i] > ema50_12h_aligned[i] and consecutive_high[i] >= 2 and 
            vol_confirm[i] and trend_filter[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < EMA50 (downtrend) AND 2 consecutive closes < S4 with volume AND trend
        elif (close[i] < ema50_12h_aligned[i] and consecutive_low[i] >= 2 and 
              vol_confirm[i] and trend_filter[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_camarilla_ema50_volume_v2"
timeframe = "4h"
leverage = 1.0