#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels from daily data with weekly trend filter and volume confirmation.
Camarilla levels identify key support/resistance levels where price often reverses or breaks out.
In trending markets, we trade breakouts in the direction of the weekly trend.
Volume confirmation ensures strong participation. Works in both bull (breakouts above R3) and bear (breakdowns below S3).
Target: 30-100 total trades over 4 years (7-25/year).
"""

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily volume SMA20 for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        if i == 0:
            continue  # Skip first bar as we need previous day
        ph = high[i-1]  # previous day high
        pl = low[i-1]   # previous day low
        pc = close[i-1] # previous day close
        
        # Camarilla levels
        R3 = pc + (ph - pl) * 1.1 / 2
        R4 = pc + (ph - pl) * 1.1
        S3 = pc - (ph - pl) * 1.1 / 2
        S4 = pc - (ph - pl) * 1.1
        
        # Volume confirmation: current volume > 1.5x average daily volume
        volume_confirm = volume[i] > 1.5 * vol_sma20_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 with weekly uptrend and volume confirmation
            if close[i] > R3 and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with weekly downtrend and volume confirmation
            elif close[i] < S3 and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters below R3 or trend reversal
            if close[i] < R3 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters above S3 or trend reversal
            if close[i] > S3 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals