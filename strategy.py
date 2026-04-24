#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA34 trend direction.
- Camarilla levels calculated from prior 1d bar: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4.
- In uptrend (price > 1w EMA34): long when price breaks above H3 with volume confirmation.
- In downtrend (price < 1w EMA34): short when price breaks below L3 with volume confirmation.
- Exit: opposite Camarilla level touch (L3 for long, H3 for short) or trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (12h).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Get 1d data for Camarilla calculation (daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from prior 1d bar
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 4
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 4
    
    # Align Camarilla levels to 12h (using prior completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # Need enough 1w bars for EMA and 1d for Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_1w_aligned[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Uptrend: price > EMA34 -> look for long breakout above H3
                if price > ema_trend and price > h3_level:
                    signals[i] = 0.25
                    position = 1
                # Downtrend: price < EMA34 -> look for short breakdown below L3
                elif price < ema_trend and price < l3_level:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price touches L3 (mean reversion) or trend reverses
            if price < l3_level or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches H3 (mean reversion) or trend reverses
            if price > h3_level or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0