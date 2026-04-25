#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray + 1d EMA50 Trend Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend phase and avoids whipsaws.
Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
Combined with 1d EMA50 trend filter to only trade in alignment with higher timeframe trend.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-30 trades/year on 6h timeframe.
Works in bull via long signals when Bull Power > 0 and price > Lips.
Works in bear via short signals when Bear Power > 0 and price < Jaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA (same as Wilder's smoothing)
    # SMMA today = (SMMA yesterday * (period-1) + price today) / period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (13 periods) and EMA13
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema13[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND price > EMA50 (1d uptrend)
            long_entry = (lips[i] > teeth[i] > jaw[i]) and (bull_power[i] > 0) and (curr_close > ema_trend)
            # Short: Jaw > Teeth > Lips (bearish alignment) AND Bear Power > 0 AND price < EMA50 (1d downtrend)
            short_entry = (jaw[i] > teeth[i] > lips[i]) and (bear_power[i] > 0) and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Alligator turns bearish (Jaw > Teeth > Lips) OR Bull Power <= 0 OR price < EMA50
            if (jaw[i] > teeth[i] > lips[i]) or (bull_power[i] <= 0) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Alligator turns bullish (Lips > Teeth > Jaw) OR Bear Power <= 0 OR price > EMA50
            if (lips[i] > teeth[i] > jaw[i]) or (bear_power[i] <= 0) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_Elder_Ray_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0