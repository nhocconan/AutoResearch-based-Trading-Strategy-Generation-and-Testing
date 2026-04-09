#!/usr/bin/env python3
# 1d_weekly_camarilla_pivot_breakout_volume_v1
# Hypothesis: 1d strategy using weekly Camarilla pivot levels (from 1w HTF) for breakout entries, volume confirmation (>1.5x 10-bar avg volume), and trend alignment via 1d EMA(50). Enters long when price breaks above weekly R4 with volume and price > 1d EMA(50); enters short when price breaks below weekly S4 with volume and price < 1d EMA(50). Exits on opposite pivot level touch (R3/S3) or close beyond R5/S5. Uses discrete sizing (0.25) to limit fee churn. Target: 7-25 trades/year (30-100 total over 4 years). Weekly pivots provide structural support/resistance that works in bull/bear markets; volume confirms breakout conviction; 1d EMA filters counter-trend noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_pivot_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (10-period = 10 days)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=10, min_periods=10).mean().values
    
    # Multi-timeframe: 1d EMA(50) trend filter (using same timeframe data)
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Multi-timeframe: weekly Camarilla pivot levels (from 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    # Camarilla pivot calculation: based on prior week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3 = pivot + (range_1w * 1.1 / 4)
    r4 = pivot + (range_1w * 1.1 / 2)
    r5 = pivot + (range_1w * 1.1)
    s3 = pivot - (range_1w * 1.1 / 4)
    s4 = pivot - (range_1w * 1.1 / 2)
    s5 = pivot - (range_1w * 1.1)
    
    # Align weekly levels to 1d timeframe (wait for weekly close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r5_aligned = align_htf_to_ltf(prices, df_1w, r5)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    s5_aligned = align_htf_to_ltf(prices, df_1w, s5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r5_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(s5_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 10-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filters
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        if position == 1:  # Long position
            # Exit: price touches weekly R3 or breaks above R5 (failed breakout)
            if close[i] <= r3_aligned[i] or close[i] >= r5_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches weekly S3 or breaks below S5 (failed breakout)
            if close[i] >= s3_aligned[i] or close[i] <= s5_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for weekly breakout with volume and trend alignment
            bullish_breakout = (close[i] > r4_aligned[i]) and volume_confirmed and uptrend
            bearish_breakout = (close[i] < s4_aligned[i]) and volume_confirmed and downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals