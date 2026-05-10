#!/usr/bin/env python3

"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Breakouts from Camarilla R1/S1 levels with 1d trend filter and volume confirmation.
Trades only in the direction of the 1d trend to avoid whipsaws. Uses volume > 2x average to confirm breakout strength.
Exit when price closes back inside the R1-S1 range (mean reversion).
Designed for 12h timeframe to capture fewer, higher-quality trades (target: 12-37/year).
Works in bull/bear by following 1d trend.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d trend (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for 12h bar (using previous bar's OHLC)
        if i == 0:
            continue
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        range_ = ph - pl
        
        # Avoid division by zero
        if range_ <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla levels
        r1 = pc + (range_ * 1.1 / 12)
        s1 = pc - (range_ * 1.1 / 12)
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        trend_up = trend_1d_up_aligned[i] > 0.5
        trend_down = trend_1d_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: break above R1 + 1d uptrend + volume
            if close[i] > r1 and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 + 1d downtrend + volume
            elif close[i] < s1 and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price closes back below R1 (mean reversion to range)
            if close[i] < r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes back above S1
            if close[i] > s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals