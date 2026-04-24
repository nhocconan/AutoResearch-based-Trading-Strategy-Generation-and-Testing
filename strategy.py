#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1-week EMA trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend direction.
- Williams %R(14) identifies overbought/oversold conditions: < -80 oversold, > -20 overbought.
- In uptrend (1w EMA50 > EMA200): look for long entries when %R crosses above -80 from below.
- In downtrend (1w EMA50 < EMA200): look for short entries when %R crosses below -20 from above.
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter weak signals.
- Exit: Opposite %R crossover or trend reversal.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in bull markets via trend-following pullsbacks and in bear markets via counter-trend bounces.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 and EMA200 for trend
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = close_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMAs to 1d
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Williams %R (14-period) on 1d
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 200, lookback, 20)  # Need enough bars for EMAs and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 1 = uptrend, -1 = downtrend
        trend = 1 if ema50_1w_aligned[i] > ema200_1w_aligned[i] else -1
        curr_wr = williams_r[i]
        prev_wr = williams_r[i-1]
        curr_close = close[i]
        curr_volume_ok = volume_spike[i]
        
        if position == 0:
            # Check for entry signals
            if curr_volume_ok:
                if trend == 1:  # Uptrend: look for long on %R crossing above -80 (oversold bounce)
                    if prev_wr <= -80 and curr_wr > -80:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend: look for short on %R crossing below -20 (overbought bounce)
                    if prev_wr >= -20 and curr_wr < -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: %R crosses below -50 (momentum loss) OR trend reverses to downtrend
            if curr_wr < -50 or trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: %R crosses above -50 (momentum loss) OR trend reverses to uptrend
            if curr_wr > -50 or trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMATrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0