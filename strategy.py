#!/usr/bin/env python3
name = "1d_Weekly_Keltner_Breakout_Trend"
timeframe = "1d"
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
    
    # Get weekly data for Keltner and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for midline
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20).mean().values
    
    # Weekly ATR10 for Keltner width
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w - low_1w, np.abs(high_1w - np.roll(close_1w, 1)))
    tr2 = np.absolute(np.roll(close_1w, 1) - low_1w)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1w[0] - low_1w[0]  # first bar
    atr_1w = pd.Series(tr).ewm(span=10, min_periods=10).mean().values
    
    # Keltner bands: EMA20 ± 2*ATR10
    upper_1w = ema_1w + 2 * atr_1w
    lower_1w = ema_1w - 2 * atr_1w
    
    # Align to daily
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Daily volume spike: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(upper_1w_aligned[i]) or 
            np.isnan(lower_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close breaks above upper Keltner band in uptrend + volume spike
            if close[i] > upper_1w_aligned[i] and close[i] > ema_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below lower Keltner band in downtrend + volume spike
            elif close[i] < lower_1w_aligned[i] and close[i] < ema_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close crosses below midline OR volatility collapse
            if close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close crosses above midline OR volatility collapse
            if close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals