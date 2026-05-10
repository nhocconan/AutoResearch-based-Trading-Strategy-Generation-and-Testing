#!/usr/bin/env python3
# 12h_ParabolicSAR_VolumeTrend
# Hypothesis: 12-hour Parabolic SAR signals combined with volume trend confirmation and 1-day EMA filter.
# Parabolic SAR provides trend-following entry/exit points; volume trend confirms momentum strength; 
# Daily EMA50 filters for higher-timeframe trend alignment to avoid counter-trend trades.
# Designed for 12h to achieve 12-37 trades/year with controlled risk in both bull and bear markets.

name = "12h_ParabolicSAR_VolumeTrend"
timeframe = "12h"
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
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume trend: 20-period average (daily)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Parabolic SAR calculation
    def calculate_psar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        n = len(high)
        psar = np.zeros(n)
        trend = np.zeros(n)  # 1 for uptrend, -1 for downtrend
        af = np.zeros(n)
        ep = np.zeros(n)  # extreme point
        
        # Initialize
        psar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, n):
            if trend[i-1] == 1:  # uptrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                # Ensure PSAR doesn't exceed prior two lows
                if i >= 2:
                    psar[i] = min(psar[i], low[i-1], low[i-2])
                
                if low[i] > psar[i]:  # continue uptrend
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                else:  # reverse to downtrend
                    trend[i] = -1
                    psar[i] = ep[i-1]  # SAR becomes prior EP
                    ep[i] = low[i]
                    af[i] = af_start
            else:  # downtrend
                psar[i] = psar[i-1] + af[i-1] * (psar[i-1] - ep[i-1])
                # Ensure PSAR doesn't fall below prior two highs
                if i >= 2:
                    psar[i] = max(psar[i], high[i-1], high[i-2])
                
                if high[i] < psar[i]:  # continue downtrend
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                else:  # reverse to uptrend
                    trend[i] = 1
                    psar[i] = ep[i-1]  # SAR becomes prior EP
                    ep[i] = high[i]
                    af[i] = af_start
        
        return psar, trend
    
    psar, psar_trend = calculate_psar(high, low)
    
    # Align daily indicators to 12h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: PSAR bullish (close above SAR), above daily EMA50, volume above average
            if close[i] > psar[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: PSAR bearish (close below SAR), below daily EMA50, volume above average
            elif close[i] < psar[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: PSAR turns bearish (close below SAR) or below daily EMA50
            if close[i] < psar[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: PSAR turns bullish (close above SAR) or above daily EMA50
            if close[i] > psar[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals