#!/usr/bin/env python3
name = "1d_KAMA_1wTrend_VolumeFilter_v1"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w KAMA trend filter (more adaptive than EMA)
    price_series = pd.Series(df_1w['close'])
    change = abs(price_series.diff(1))
    volatility = price_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = [price_series.iloc[0]]
    for i in range(1, len(price_series)):
        kama.append(kama[-1] + sc.iloc[i] * (price_series.iloc[i] - kama[-1]))
    kama = np.array(kama)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily KAMA for entry signal
    price_series_d = pd.Series(close)
    change_d = abs(price_series_d.diff(1))
    volatility_d = price_series_d.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er_d = change_d / volatility_d.replace(0, np.nan)
    er_d = er_d.fillna(0)
    sc_d = (er_d * (0.6645 - 0.0645) + 0.0645) ** 2
    kama_d = [price_series_d.iloc[0]]
    for i in range(1, len(price_series_d)):
        kama_d.append(kama_d[-1] + sc_d.iloc[i] * (price_series_d.iloc[i] - kama_d[-1]))
    kama_d = np.array(kama_d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Need 20 for volume MA, 30 for weekly KAMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(kama_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily KAMA AND above weekly KAMA + volume
            if close[i] > kama_d[i] and close[i] > kama_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below daily KAMA AND below weekly KAMA + volume
            elif close[i] < kama_d[i] and close[i] < kama_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses daily KAMA in opposite direction
            if position == 1:
                if close[i] < kama_d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals