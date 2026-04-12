#!/usr/bin/env python3
"""
1d_1w_keltner_breakout_volume_trend_v1
Hypothesis: Daily strategy using weekly Keltner channel breakout with volume confirmation and weekly trend filter.
Enters long when price breaks above upper Keltner band with volume spike and weekly uptrend; short when breaks below lower band with volume spike and weekly downtrend.
Uses fixed position size (0.25) to minimize trading frequency and fee drag.
Designed to capture strong trends while avoiding choppy markets via weekly trend filter.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag while capturing strong moves.
"""

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
    
    # Get weekly data for Keltner channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for Keltner center
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR(10) for Keltner width
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(np.roll(high_1w, 1) - close_1w)
    tr3 = np.abs(np.roll(low_1w, 1) - close_1w)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly Keltner bands
    keltner_upper = ema20_1w + 2.0 * atr_1w
    keltner_lower = ema20_1w - 2.0 * atr_1w
    
    # Align weekly Keltner bands to daily timeframe
    keltner_upper_daily = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_daily = align_htf_to_ltf(prices, df_1w, keltner_lower)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly trend filter: price above/below EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper_daily[i]) or np.isnan(keltner_lower_daily[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Trend filter from weekly EMA50
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions: Keltner breakout with volume and trend confirmation
        long_breakout = close[i] > keltner_upper_daily[i] and volume_filter and uptrend_1w
        short_breakout = close[i] < keltner_lower_daily[i] and volume_filter and downtrend_1w
        
        # Exit conditions: reverse back to weekly EMA20 or trend change
        long_exit = close[i] < ema20_1w_aligned[i] or not uptrend_1w
        short_exit = close[i] > ema20_1w_aligned[i] or not downtrend_1w
        
        # Fixed position size to minimize trading frequency
        position_size = 0.25
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_keltner_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0