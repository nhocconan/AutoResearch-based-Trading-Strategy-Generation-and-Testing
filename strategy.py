#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h KAMA trend with 1d RSI filter and volume spike
    # KAMA adapts to market noise - fast in trends, slow in ranges
    # Combines with 1d RSI (overbought/oversold) and volume confirmation
    # Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, slow=2, fast=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_val = np.zeros_like(close)
        kama_val[0] = close[0]
        for i in range(1, len(close)):
            kama_val[i] = kama_val[i-1] + sc[i] * (close[i] - kama_val[i-1])
        return kama_val
    
    kama_val = kama(close)
    
    # 1d RSI filter
    df_1d = get_htf_data(prices, '1d')
    rsi_14 = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        up = np.maximum(df_1d['close'].iloc[i] - df_1d['close'].iloc[i-1], 0)
        down = np.maximum(df_1d['close'].iloc[i-1] - df_1d['close'].iloc[i], 0)
        # Simplified RSI calculation using Wilder's smoothing
        if i == 14:
            avg_up = np.mean(df_1d['close'].iloc[1:15] - df_1d['close'].iloc[0:14])
            avg_down = np.mean(df_1d['close'].iloc[0:14] - df_1d['close'].iloc[1:15])
            avg_up = np.where(avg_up < 0, 0, avg_up)
            avg_down = np.where(avg_down < 0, 0, avg_down)
        else:
            avg_up = (avg_up * 13 + up) / 14
            avg_down = (avg_down * 13 + down) / 14
        rs = avg_up / avg_down if avg_down != 0 else 0
        rsi_14[i] = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Volume spike
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(30, n):
        if np.isnan(kama_val[i]) or np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA (uptrend) + RSI < 30 (oversold) + volume spike
            if close[i] > kama_val[i] and rsi_14_aligned[i] < 30 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (downtrend) + RSI > 70 (overbought) + volume spike
            elif close[i] < kama_val[i] and rsi_14_aligned[i] > 70 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend change or RSI mean reversion
            if position == 1:
                if close[i] < kama_val[i] or rsi_14_aligned[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_val[i] or rsi_14_aligned[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_1dRSI_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0