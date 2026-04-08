# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_1d_volume_breakout_v1
Hypothesis: 6h breakout of 20-bar high/low with volume confirmation and 1d trend filter.
Long when price breaks above 6h high(20) with volume > 1.5x avg and 1d bullish trend (close > EMA50).
Short when price breaks below 6h low(20) with volume > 1.5x avg and 1d bearish trend (close < EMA50).
Designed to capture strong momentum moves with institutional volume in both bull and bear markets.
Target: 12-30 trades/year per symbol (48-120 total over 4 years) by requiring volume and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_volume_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend bias: close > EMA(50) for bullish, close < EMA(50) for bearish
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish = close_1d > ema_50
    trend_bearish = close_1d < ema_50
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    # 6h range calculation: highest high and lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=20, min_periods=20).max().values
    lowest_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 6h low(20) or 1d trend turns bearish
            if close[i] < lowest_low[i] or trend_bearish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above 6h high(20) or 1d trend turns bullish
            if close[i] > highest_high[i] or trend_bullish_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 6h high(20) with volume and 1d bullish trend
            if close[i] > highest_high[i] and vol_confirm[i] and trend_bullish_aligned[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 6h low(20) with volume and 1d bearish trend
            elif close[i] < lowest_low[i] and vol_confirm[i] and trend_bearish_aligned[i] > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals