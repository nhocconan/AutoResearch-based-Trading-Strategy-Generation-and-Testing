#!/usr/bin/env python3
name = "1d_WeeklyKeltnerBreakout_TrendFilter_Volume"
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
    
    # Weekly trend: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly ATR for Keltner channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[np.nan], close_1w[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1w = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Keltner channels (weekly)
    keltner_upper = ema_50_1w_aligned + 2.0 * atr_1w_aligned
    keltner_lower = ema_50_1w_aligned - 2.0 * atr_1w_aligned
    
    # Daily volume spike: volume > 2.0 * 20-day SMA of volume
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_sma
    
    # Daily trend filter: EMA50 on daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Keltner upper + daily EMA50 + volume spike
            if close[i] > keltner_upper[i] and close[i] > ema_50_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Keltner lower + daily EMA50 + volume spike
            elif close[i] < keltner_lower[i] and close[i] < ema_50_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA50
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above daily EMA50
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals