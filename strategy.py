#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike and 1d EMA34 trend filter.
- Primary timeframe: 4h for execution, HTF: 1d for Camarilla pivots, volume MA, and EMA trend.
- Volume confirmation: current volume > 1.5 * 20-period volume MA on 1d (avoid false breakouts).
- Trend filter: price > 1d EMA34 for bullish bias, price < 1d EMA34 for bearish bias.
- Entry: Long when price breaks above H3 AND price > EMA34 (bullish breakout in uptrend).
         Short when price breaks below L3 AND price < EMA34 (bearish breakout in downtrend).
- Exit: Opposite Camarilla breakout (price touches L3 for long exit, H3 for short exit).
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
    
    # Get 1d data for Camarilla pivots, volume MA, and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) on 1d
    # Typical Price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    # Camarilla width = (H - L) * 1.1 / 8
    width = (df_1d['high'] - df_1d['low']) * 1.1 / 8.0
    # H3 = C + width * 1.1
    camarilla_h3 = df_1d['close'].values + width * 1.1
    # L3 = C - width * 1.1
    camarilla_l3 = df_1d['close'].values - width * 1.1
    
    # Calculate 20-period volume MA on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA34 on 1d close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current 4h volume > 1.5 * aligned 1d volume MA
    volume_spike = volume > (1.5 * volume_ma_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # Need enough 1d bars for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        ema34 = ema34_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish breakout: price closes above H3 AND above EMA34 (uptrend)
                if curr_close > h3 and curr_close > ema34:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below L3 AND below EMA34 (downtrend)
                elif curr_close < l3 and curr_close < ema34:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below L3 (opposite Camarilla level)
            if curr_close < l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 (opposite Camarilla level)
            if curr_close > h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dVolumeSpike_1dEMA34Trend_v1"
timeframe = "4h"
leverage = 1.0