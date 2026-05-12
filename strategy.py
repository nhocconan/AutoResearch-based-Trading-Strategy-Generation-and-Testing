#!/usr/bin/env python3
name = "1h_4hTrend_1dVolatility_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend: EMA21 on 4h close
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 1d volatility: ATR(14) normalized by close
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_norm_1d = atr14_1d / close_1d
    atr_norm_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_norm_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 21  # ensure EMA21 has enough data
    
    for i in range(start_idx, n):
        if np.isnan(ema21_4h_aligned[i]) or np.isnan(atr_norm_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            if in_session:
                # Long: above 4h EMA21 + low volatility
                if close[i] > ema21_4h_aligned[i] and atr_norm_1d_aligned[i] < 0.02:
                    signals[i] = 0.20
                    position = 1
                # Short: below 4h EMA21 + low volatility
                elif close[i] < ema21_4h_aligned[i] and atr_norm_1d_aligned[i] < 0.02:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: below 4h EMA21 or high volatility
            if close[i] < ema21_4h_aligned[i] or atr_norm_1d_aligned[i] >= 0.025:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: above 4h EMA21 or high volatility
            if close[i] > ema21_4h_aligned[i] or atr_norm_1d_aligned[i] >= 0.025:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals