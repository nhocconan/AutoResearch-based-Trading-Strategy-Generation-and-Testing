#!/usr/bin/env python3
# 4h_Keltner_Breakout_Volume_Trend
# Hypothesis: Keltner Channel breakouts with volume confirmation and 1d EMA trend filter provide high-probability entries.
# Works in bull markets via upper band breakouts and in bear markets via lower band breakdowns.
# Uses discrete position sizing (0.25) to limit overtrading and fee drag. Targets 20-40 trades/year.

name = "4h_Keltner_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20-period EMA, 2x ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get 1d close for trend determination
        close_1d_series = pd.Series(close_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_series.values)
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Keltner upper in uptrend with volume
            if is_uptrend and close[i] > kc_upper[i] and volume_condition:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Keltner lower in downtrend with volume
            elif is_downtrend and close[i] < kc_lower[i] and volume_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to EMA20 or trend turns down
            if close[i] < ema_20[i] or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to EMA20 or trend turns up
            if close[i] > ema_20[i] or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals