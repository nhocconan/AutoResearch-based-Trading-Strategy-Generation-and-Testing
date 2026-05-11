#!/usr/bin/env python3
name = "6h_Keltner_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Keltner Channel: EMA(20) ± 2*ATR(10)
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    ema_kc = close_series.ewm(span=20, adjust=False, min_periods=20).mean()
    
    # True Range
    tr1 = high_series - low_series
    tr2 = np.abs(high_series - close_series.shift())
    tr3 = np.abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=10, adjust=False, min_periods=10).mean()
    
    upper_keltner = ema_kc + 2 * atr
    lower_keltner = ema_kc - 2 * atr
    
    upper_keltner_vals = upper_keltner.values
    lower_keltner_vals = lower_keltner.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(upper_keltner_vals[i]) or
            np.isnan(lower_keltner_vals[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > upper Keltner + 1d uptrend + volume spike
            if close[i] > upper_keltner_vals[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Keltner + 1d downtrend + volume spike
            elif close[i] < lower_keltner_vals[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < EMA(20) or 1d trend down
            if close[i] < ema_kc.iloc[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > EMA(20) or 1d trend up
            if close[i] > ema_kc.iloc[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals