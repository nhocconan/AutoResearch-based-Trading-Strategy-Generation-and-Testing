#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_FVG_Filter
Hypothesis: Camarilla R3/S3 breakouts on 6h with 1d EMA50 trend filter and Fair Value Gap (FVG) confirmation.
R3/S3 represent stronger Camarilla levels - breakouts with trend and FVG (liquidity void) have high continuation probability in both bull and bear markets.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    # R3 = Close + (High-Low)*1.1/4
    # S3 = Close - (High-Low)*1.1/4
    camarilla_r3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align HTF indicators to 6h timeframe (completed 1d bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d, additional_delay_bars=1)
    
    # Fair Value Gap (FVG) detection on 6h timeframe
    # Bullish FVG: low[i] > high[i-2] (gap up)
    # Bearish FVG: high[i] < low[i-2] (gap down)
    bullish_fvg = np.zeros(n, dtype=bool)
    bearish_fvg = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        if low[i] > high[i-2]:
            bullish_fvg[i] = True
        if high[i] < low[i-2]:
            bearish_fvg[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and FVG
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1d trend with FVG confirmation
            # Long: price breaks above R3 in uptrend (close > EMA50) with bullish FVG
            # Short: price breaks below S3 in downtrend (close < EMA50) with bearish FVG
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema50_aligned[i]) and bullish_fvg[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema50_aligned[i]) and bearish_fvg[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla R3 (mean reversion at breakout level)
            exit_signal = close[i] < camarilla_r3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla S3 (mean reversion at breakout level)
            exit_signal = close[i] > camarilla_s3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_FVG_Filter"
timeframe = "6h"
leverage = 1.0