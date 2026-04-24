#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend direction.
- EMA50 > EMA50[5] indicates bullish weekly trend (5-day slope), EMA50 < EMA50[5] indicates bearish.
- In bullish weekly trend: Long when price breaks above H3 level with volume spike.
- In bearish weekly trend: Short when price breaks below L3 level with volume spike.
- Exit: Opposite Camarilla break (L3 for long, H3 for short) or weekly trend flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
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
    
    # Get 1w data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Calculate Camarilla levels (H3, L3) from previous 1d bar
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # Using previous bar's OHLC to avoid look-ahead
    prev_close = pd.Series(close).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(55, 20)  # Need enough 1w bars for EMA50 and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Weekly trend: EMA50 slope over 5 bars (approx 5 weeks)
        if i >= 5:
            ema50_now = ema50_aligned[i]
            ema50_prev = ema50_aligned[i-5]
            weekly_bullish = ema50_now > ema50_prev
            weekly_bearish = ema50_now < ema50_prev
        else:
            weekly_bullish = False
            weekly_bearish = False
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if weekly_bullish:
                    # Bullish weekly trend: long on H3 breakout
                    if curr_high > camarilla_h3[i]:
                        signals[i] = 0.25
                        position = 1
                elif weekly_bearish:
                    # Bearish weekly trend: short on L3 breakdown
                    if curr_low < camarilla_l3[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR weekly trend turns bearish
            if curr_low < camarilla_l3[i] or (i >= 5 and ema50_aligned[i] < ema50_aligned[i-5]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR weekly trend turns bullish
            if curr_high > camarilla_h3[i] or (i >= 5 and ema50_aligned[i] > ema50_aligned[i-5]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0