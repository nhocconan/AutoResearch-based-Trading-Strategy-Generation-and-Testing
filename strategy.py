#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator strategy with 1w EMA200 trend filter and volume confirmation.
Williams Alligator (Jaw=TEETH=LIPS SMMA) identifies trending vs ranging markets.
In trending markets (JAW > TEETH > LIPS for uptrend, reverse for downtrend), we enter on
Alligator "awakening" (lines diverging) with volume confirmation and 1w trend alignment.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 12h timeframe
to target 12-37 trades/year (50-150 total over 4 years) and work in both bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA200 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    jaw = smma(median_12h, 13)
    teeth = smma(median_12h, 8)
    lips = smma(median_12h, 5)
    
    # Align Alligator lines to primary timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate volume MA (30-period) for confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30)  # need Alligator and volume MA30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1w trend filter: close > EMA200 = uptrend bias, close < EMA200 = downtrend bias
        trend_up = close[i] > ema_200_1w_aligned[i]
        trend_down = close[i] < ema_200_1w_aligned[i]
        
        # Volume filter: 12h volume > 1.8x 30-period MA
        vol_filter = volume[i] > 1.8 * vol_ma_30[i]
        
        # Williams Alligator signals:
        # Uptrend alignment: JAW > TEETH > LIPS (alligator eating with mouth up)
        # Downtrend alignment: JAW < TEETH < LIPS (alligator eating with mouth down)
        # Additionally, we require some separation (awakening) to avoid chop
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Calculate Alligator separation (minimum distance between lines)
        separation = min(abs(jaw_val - teeth_val), abs(teeth_val - lips_val), abs(jaw_val - lips_val))
        # Require minimum separation to avoid choppy markets (adaptive to price level)
        min_separation = close[i] * 0.001  # 0.1% of price
        alligator_awake = separation > min_separation
        
        alligator_uptrend = (jaw_val > teeth_val) and (teeth_val > lips_val)
        alligator_downtrend = (jaw_val < teeth_val) and (teeth_val < lips_val)
        
        if position == 0:
            # Long: Alligator uptrend + awake + 1w uptrend bias + volume confirmation
            if alligator_uptrend and alligator_awake and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + awake + 1w downtrend bias + volume confirmation
            elif alligator_downtrend and alligator_awake and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator changes direction or goes to sleep (lines intertwine)
            exit_signal = False
            if position == 1:
                # Exit long if Alligator no longer in uptrend alignment or falls asleep
                if not (alligator_uptrend and alligator_awake):
                    exit_signal = True
            elif position == -1:
                # Exit short if Alligator no longer in downtrend alignment or falls asleep
                if not (alligator_downtrend and alligator_awake):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1wEMA200_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0