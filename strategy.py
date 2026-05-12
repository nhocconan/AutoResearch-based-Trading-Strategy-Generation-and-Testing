#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_12H_TREND_FILTER
# Hypothesis: Camarilla R1/S1 breakouts capture short-term reversals with high probability.
# In 12h uptrend (price > EMA50), go long when price breaks above R1; in downtrend, go short when price breaks below S1.
# Uses volume confirmation (>1.5x average volume) to avoid false breakouts.
# Works in both bull and bear markets: trend filter avoids counter-trend trades, Camarilla levels provide precise entry/exit.
# Target: 20-35 trades/year on 4h timeframe.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_12H_TREND_FILTER"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # EMA50 for 12h trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Daily data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 12h uptrend + price breaks above R1 + volume confirmation
            if (close[i] > ema50_12h_aligned[i] and 
                close[i] > r1_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 12h downtrend + price breaks below S1 + volume confirmation
            elif (close[i] < ema50_12h_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price breaks below S1 (mean reversion)
            if (close[i] <= ema50_12h_aligned[i] or 
                close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price breaks above R1 (mean reversion)
            if (close[i] >= ema50_12h_aligned[i] or 
                close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals