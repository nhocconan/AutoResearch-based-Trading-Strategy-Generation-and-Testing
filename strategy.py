#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend direction.
- EMA34 > EMA89 indicates bullish trend (favor long breakouts at R1/S1).
- EMA34 < EMA89 indicates bearish trend (favor short breakouts at R1/S1).
- Entry: Long when price breaks above Camarilla R1 AND EMA34 > EMA89 (bullish breakout in uptrend).
         Short when price breaks below Camarilla S1 AND EMA34 < EMA89 (bearish breakout in downtrend).
- Exit: Opposite Camarilla breakout (S1 for longs, R1 for shorts) or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:
        return np.zeros(n)
    
    # Calculate EMA34 and EMA89 on 1d
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_1d = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d EMAs to 6h
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema89_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla levels: based on previous day's high, low, close
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(90, 20)  # Need enough 1d bars for EMA89 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema89_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                ema34_val = ema34_aligned[i]
                ema89_val = ema89_aligned[i]
                
                # Bullish breakout: price closes above R1 AND EMA34 > EMA89 (uptrend)
                if curr_close > r1_aligned[i] and ema34_val > ema89_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below S1 AND EMA34 < EMA89 (downtrend)
                elif curr_close < s1_aligned[i] and ema34_val < ema89_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below S1 OR EMA trend turns bearish
            if curr_close < s1_aligned[i] or ema34_aligned[i] < ema89_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 OR EMA trend turns bullish
            if curr_close > r1_aligned[i] or ema34_aligned[i] > ema89_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1dEMATrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0