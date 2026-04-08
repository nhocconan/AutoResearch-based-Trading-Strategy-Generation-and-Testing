#!/usr/bin/env python3
"""
12h_momentum_breakout_volume_v1
Hypothesis: Use 12h price momentum (close > open) with 1d volume confirmation and 1w trend filter.
Enter long when 12h candle is bullish, volume > 1.5x 24-period average, and 1w EMA(21) is rising.
Enter short when 12h candle is bearish, volume > 1.5x 24-period average, and 1w EMA(21) is falling.
Uses 12h timeframe for entries, 1w for trend filter. Target: 15-25 trades/year.
Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_momentum_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA(21) for trend
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # 12h candle direction
        candle_bullish = close[i] > open_price[i]
        candle_bearish = close[i] < open_price[i]
        
        if position == 1:  # Long position
            # Exit: candle turns bearish or 1w trend turns down
            if not candle_bullish or ema_1w_aligned[i] < ema_1w_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: candle turns bullish or 1w trend turns up
            if not candle_bearish or ema_1w_aligned[i] > ema_1w_aligned[max(0, i-1)]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: bullish 12h candle with volume and 1w uptrend
            if candle_bullish and vol_confirm[i] and ema_1w_aligned[i] > ema_1w_aligned[max(0, i-1)]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish 12h candle with volume and 1w downtrend
            elif candle_bearish and vol_confirm[i] and ema_1w_aligned[i] < ema_1w_aligned[max(0, i-1)]:
                position = -1
                signals[i] = -0.25
    
    return signals