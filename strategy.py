#!/usr/bin/env python3
name = "12h_KAMA_Direction_1wTrend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # KAMA direction on 12h
    close_s = pd.Series(close)
    delta = close_s.diff().abs()
    vol = close_s.rolling(window=10, min_periods=10).sum()
    er = delta / vol.replace(0, np.nan)
    er = er.fillna(1).values
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: volume > 1.5 * average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if 1w trend data not ready
        if np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + volume confirmation + 1w uptrend
            if close[i] > kama[i] and vol_confirm[i] and (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + volume confirmation + 1w downtrend
            elif close[i] < kama[i] and vol_confirm[i] and (close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or trend reversal
            if close[i] < kama[i] or (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or trend reversal
            if close[i] > kama[i] or (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals