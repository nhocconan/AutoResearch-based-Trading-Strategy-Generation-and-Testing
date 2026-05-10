#!/usr/bin/env python3
"""
4h_Parabolic_SAR_EMA_Trend
Hypothesis: Parabolic SAR provides clear trend direction with built-in trailing stop.
In trending markets (determined by EMA50), price respects SAR levels.
Long when price > SAR and price > EMA50 with volume confirmation.
Short when price < SAR and price < EMA50 with volume confirmation.
Exit when price crosses SAR (trend reversal) or volume confirmation fails.
Designed for 4h timeframe to capture intermediate trends with minimal whipsaw.
Targets 80-150 trades over 4 years (20-37/year) to balance opportunity and cost.
Works in bull (follow SAR uptrend) and bear (follow SAR downtrend).
"""

name = "4h_Parabolic_SAR_EMA_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Parabolic SAR calculation
    # Start with long assumption
    psar = np.zeros(n)
    psar[0] = low[0]
    bull = True  # True for long, False for short
    af = 0.02    # acceleration factor
    max_af = 0.2
    ep = high[0] if bull else low[0]  # extreme point
    
    for i in range(1, n):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't penetrate previous two lows
            if i >= 2:
                psar[i] = min(psar[i], low[i-1], low[i-2])
            # Check for trend reversal
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep  # SAR becomes previous EP
                af = 0.02
                ep = low[i]
            else:
                # Continue trend
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + 0.02, max_af)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR doesn't penetrate previous two highs
            if i >= 2:
                psar[i] = max(psar[i], high[i-1], high[i-2])
            # Check for trend reversal
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep  # SAR becomes previous EP
                af = 0.02
                ep = high[i]
            else:
                # Continue trend
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + 0.02, max_af)
    
    # EMA50 for trend filter (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 4h volume > 1.5x average 4h volume (using 1d SMA20 scaled)
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = alpha * volume_1d[i] + (1 - alpha) * vol_sma20_1d[i-1]
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    # Scale daily volume average to 4h (6 periods of 4h in 1d)
    vol_4h_avg = vol_sma20_1d_aligned / 6.0
    volume_confirm = volume > 1.5 * vol_4h_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need EMA50 warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_4h_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above SAR and above EMA50 with volume confirmation
            if close[i] > psar[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below SAR and below EMA50 with volume confirmation
            elif close[i] < psar[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long: Continue if price above SAR and EMA50, else exit
            if close[i] > psar[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
            else:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: Continue if price below SAR and EMA50, else exit
            if close[i] < psar[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals