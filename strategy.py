#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (EMA50) and volume confirmation.
# Long when price breaks above R1 (resistance) AND close > 4h EMA50 AND volume > 1.3x average.
# Short when price breaks below S1 (support) AND close < 4h EMA50 AND volume > 1.3x average.
# Exit when price reverts to Camarilla pivot (PP) or trend reverses.
# Uses 4h for signal direction (EMA50), 1h only for precise entry timing via Camarilla levels.
# Session filter: 08-20 UTC to avoid low-volume noise. Target: 60-150 total trades over 4 years.
# Works in bull via breakout continuation, bear via faded rallies and mean reversion to pivot.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla calculation (using previous bar's OHLC)
    df_1h = prices.copy()
    df_1h['prev_close'] = df_1h['close'].shift(1)
    df_1h['prev_high'] = df_1h['high'].shift(1)
    df_1h['prev_low'] = df_1h['low'].shift(1)
    
    # Camarilla levels based on previous bar
    PP = (df_1h['prev_high'] + df_1h['prev_low'] + df_1h['prev_close']) / 3
    R1 = PP + (df_1h['prev_high'] - df_1h['prev_low']) * 1.1 / 12
    S1 = PP - (df_1h['prev_high'] - df_1h['prev_low']) * 1.1 / 12
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current 1h volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN or outside session
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(PP[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 AND close > 4h EMA50 AND volume confirmation
            if close[i] > R1[i] and close[i] > ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S1 AND close < 4h EMA50 AND volume confirmation
            elif close[i] < S1[i] and close[i] < ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reverts to pivot (PP) OR trend reversal (close < 4h EMA50)
            if close[i] <= PP[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price reverts to pivot (PP) OR trend reversal (close > 4h EMA50)
            if close[i] >= PP[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals