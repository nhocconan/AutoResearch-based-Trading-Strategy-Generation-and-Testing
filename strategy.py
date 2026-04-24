#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- EMA > price = bullish bias (only look for longs), EMA < price = bearish bias (only look for shorts).
- Entry: Long when price breaks above Camarilla H3 level AND EMA12h > price (bullish breakout in uptrend).
         Short when price breaks below Camarilla L3 level AND EMA12h < price (bearish breakout in downtrend).
- Exit: Opposite Camarilla breakout (H3 for shorts, L3 for longs) or EMA cross reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why it should work: Camarilla levels identify institutional support/resistance; EMA filter ensures we trade with the higher timeframe trend; volume confirmation reduces false signals. Works in both bull and bear markets by adapting direction based on 12h trend.
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
    
    # Get 12h data for EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate EMA (34-period) on 12h
    ema_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Camarilla pivot levels (H3, L3) from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels use previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla H3 and L3
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align 1d Camarilla levels to 6h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need enough 12h bars for EMA, 20 for volume MA, 1 for Camarilla (shifted)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_12h_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price breaks above H3 AND EMA > price (uptrend bias)
                if curr_high > h3_level and ema_val > curr_close:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 AND EMA < price (downtrend bias)
                elif curr_low < l3_level and ema_val < curr_close:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA cross below price (trend reversal)
            if curr_low < l3_level or ema_val < curr_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA cross above price (trend reversal)
            if curr_high > h3_level or ema_val > curr_close:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0