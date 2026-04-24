#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA (12h) to confirm breakout strength.
- Entry: Long when price breaks above Camarilla H3 level AND 1d EMA34 trend bullish AND volume spike.
         Short when price breaks below Camarilla L3 level AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Why should work in both bull/bear: EMA34 filter ensures we only trade with the 1d trend,
  reducing false breakouts in sideways markets. Volume spike confirms institutional participation.
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
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # We need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivots: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels: H3 = pivot + (range * 1.1/4), L3 = pivot - (range * 1.1/4)
    camarilla_h3 = pivot + (range_val * 1.1 / 4.0)
    camarilla_l3 = pivot - (range_val * 1.1 / 4.0)
    
    # Align daily Camarilla levels to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 12h data for EMA(34) trend and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h close
    ema_34 = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 12h bars for EMA34 and volume MA, plus 1d for pivots
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above H3 AND 12h EMA34 bullish (price > EMA34)
                if curr_high > h3_level and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 AND 12h EMA34 bearish (price < EMA34)
                elif curr_low < l3_level and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume confirmation
            if curr_low < l3_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume confirmation
            if curr_high > h3_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0