#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_With_Trend_Filter
Hypothesis: Camarilla pivot levels (S1, S2, S3, R1, R2, R3) from 1d price action act as strong support/resistance.
Trade long when price crosses above R1 with 1d uptrend (close > EMA50), short when below S1 with 1d downtrend (close < EMA50).
Volume confirmation (volume > 1.5x 20-period average) filters false breakouts.
Designed for low frequency (10-30 trades/year) to minimize fee drag on 12h timeframe.
Works in bull (buying breakouts) and bear (selling breakdowns) markets.
"""

name = "12h_Camarilla_Pivot_With_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels ---
    # Calculated from previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    r2 = pp + (range_1d * 1.1 / 6)
    s2 = pp - (range_1d * 1.1 / 6)
    r3 = pp + (range_1d * 1.1 / 4)
    s3 = pp - (range_1d * 1.1 / 4)
    
    # Align 1d levels to 12h (wait for 1d bar to close)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # --- 1d Trend Filter: EMA50 ---
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Volume Spike (12h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema50_12h[i]):
            if position != 0:
                # Hold position until exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions
        # Long: price crosses above R1 with 1d uptrend and volume spike
        long_breakout = close[i] > r1_12h[i] and close[i-1] <= r1_12h[i-1]
        long_trend = close[i] > ema50_12h[i]
        
        # Short: price crosses below S1 with 1d downtrend and volume spike
        short_breakout = close[i] < s1_12h[i] and close[i-1] >= s1_12h[i-1]
        short_trend = close[i] < ema50_12h[i]
        
        if position == 0:
            if long_breakout and long_trend and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and short_trend and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit if price returns below R1 or trend changes
                if close[i] < r1_12h[i] or close[i] < ema50_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if price returns above S1 or trend changes
                if close[i] > s1_12h[i] or close[i] > ema50_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals