#!/usr/bin/env python3
"""
6h_Camarilla_H3L3_MeanReversion_1dTrendFilter
Hypothesis: On 6h timeframe, fade at Camarilla H3/L3 levels only when 1d EMA50 confirms the trend direction (long above EMA50, short below). This avoids counter-trend mean reversion in strong trends while capturing reversals in ranging/weak trending markets. Uses volume confirmation (>1.3x 20-bar average) and a minimum holding period of 3 bars to reduce churn. Designed for low trade frequency (target: 50-120 total trades over 4 years) to minimize fee drag. Works in bull markets (long bias above EMA50) and bear markets (short bias below EMA50) by aligning with higher timeframe trend.
"""

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
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H3 = prev_close + 1.125 * prev_range
    L3 = prev_close - 1.125 * prev_range
    
    # Align 1d pivot levels to 6h timeframe (waits for completed 1d bar)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Start index: need enough for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        ema50_val = ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals only in direction of 1d trend
            # Long: price <= L3 and above 1d EMA50 (uptrend)
            long_entry = (curr_close <= L3_aligned[i]) and (curr_close > ema50_val) and volume_confirmed[i]
            # Short: price >= H3 and below 1d EMA50 (downtrend)
            short_entry = (curr_close >= H3_aligned[i]) and (curr_close < ema50_val) and volume_confirmed[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold position for minimum 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                # Exit when price crosses 1d EMA50 (trend change) or reaches opposite Camarilla level
                if position == 1:
                    if (curr_close >= ema50_val) or (curr_close >= H3_aligned[i]):
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # position == -1
                    if (curr_close <= ema50_val) or (curr_close <= L3_aligned[i]):
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_MeanReversion_1dTrendFilter"
timeframe = "6h"
leverage = 1.0