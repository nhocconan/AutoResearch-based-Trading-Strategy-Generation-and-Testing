#!/usr/bin/env python3
# 4h_KeltnerBreakout_TrendVolume
# Hypothesis: Use Keltner Channel breakout on 4h with 20-period ATR multiplier 2.0, filtered by 1h EMA50 trend and volume spike.
# Enter long on upper band breakout with EMA50 up and volume > 1.5x average. Enter short on lower band breakout with EMA50 down and volume spike.
# Exit on opposite band touch or trend failure. Designed for 20-40 trades/year to avoid fee drag.
# Works in bull (catch breakouts) and bear (catch breakdowns) with trend filter and volume confirmation.

name = "4h_KeltnerBreakout_TrendVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def keltner_channels(high, low, close, atr_period=20, multiplier=2.0):
    """
    Calculate Keltner Channel: upper, middle (EMA), lower.
    """
    # True Range
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # EMA (middle line)
    ema = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Upper and Lower bands
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, ema, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for EMA50 trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    
    # Calculate Keltner on 4h data
    upper, middle, lower = keltner_channels(high, low, close)
    
    # 1h EMA50 for trend filter
    ema_50_1h = pd.Series(close_1h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1h data to 4h timeframe
    ema_50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_50_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_1h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1h EMA50
        trend_up = close[i] > ema_50_1h_aligned[i]
        trend_down = close[i] < ema_50_1h_aligned[i]
        
        # Volume filter: volume spike > 1.5x average
        vol_ok = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: price breaks above upper Keltner band with trend up and volume spike
            if close[i] > upper[i] and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Keltner band with trend down and volume spike
            elif close[i] < lower[i] and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price touches middle line or trend fails
            if close[i] < middle[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches middle line or trend fails
            if close[i] > middle[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals