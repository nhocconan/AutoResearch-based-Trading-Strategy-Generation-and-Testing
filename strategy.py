#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend direction.
- EMA34 > EMA50 indicates bullish trend, EMA34 < EMA50 indicates bearish trend.
- Entry: Long when price breaks above Camarilla R1 AND EMA34 > EMA50 (bullish breakout in uptrend).
         Short when price breaks below Camarilla S1 AND EMA34 < EMA50 (bearish breakout in downtrend).
- Exit: Opposite Camarilla breakout (touch S1 for long, R1 for short) or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMAs (34 and 50) on 1d
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMAs to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate previous day's Camarilla pivot levels (R1, S1) on 1d
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = pd.Series(df_1d['close']).shift(1).values
    prev_high = pd.Series(df_1d['high']).shift(1).values
    prev_low = pd.Series(df_1d['low']).shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1d bars for EMAs and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        ema_50_val = ema_50_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price breaks above R1 AND EMA34 > EMA50 (uptrend)
                if curr_high > camarilla_r1_val and ema_34_val > ema_50_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S1 AND EMA34 < EMA50 (downtrend)
                elif curr_low < camarilla_s1_val and ema_34_val < ema_50_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price touches S1 OR EMA trend reverses to downtrend
            if curr_low <= camarilla_s1_val or ema_34_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches R1 OR EMA trend reverses to uptrend
            if curr_high >= camarilla_r1_val or ema_34_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0