#!/usr/bin/env python3
"""
4h_PivotPoint_R1S1_Breakout_Volume_Confirmation_V1
Hypothesis: Camarilla pivot levels (R1, S1) act as strong intraday support/resistance.
Breakout above R1 with volume confirmation indicates bullish momentum; breakdown below S1 with volume indicates bearish momentum.
Works in bull/bear markets by only taking breakouts in direction of 12h trend (EMA34).
Volume filter ensures breakouts have conviction, reducing false signals.
Target: 20-40 trades/year per symbol with disciplined entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1, close  # pivot not used directly

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if EMA not ready
        if np.isnan(ema_34_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous bar's OHLC
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        prev_close = prices['close'].iloc[i-1]
        
        r1, s1, _ = calculate_camarilla(prev_high, prev_low, prev_close)
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > EMA34 for long, price < EMA34 for short
        trend_long = price > ema_34_12h_aligned[i]
        trend_short = price < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + uptrend
            if price > r1 and volume_ok and trend_long:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + downtrend
            elif price < s1 and volume_ok and trend_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns bearish
            if price < s1 or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns bullish
            if price > r1 or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PivotPoint_R1S1_Breakout_Volume_Confirmation_V1"
timeframe = "4h"
leverage = 1.0