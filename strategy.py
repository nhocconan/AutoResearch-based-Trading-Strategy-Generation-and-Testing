#!/usr/bin/env python3
name = "12h_1w_TRIX_Volume_ZLEMA_Trend"
timeframe = "12h"
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
    
    # 1W ZLEMA(9) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # ZLEMA calculation: EMA of [2*close - lag(close, period)]
    lag = 9
    dema_1w = np.zeros_like(close_1w)
    dema_1w[:] = np.nan
    ema1 = np.zeros_like(close_1w)
    ema1[:] = np.nan
    ema2 = np.zeros_like(close_1w)
    ema2[:] = np.nan
    
    # First EMA
    alpha = 2.0 / (lag + 1)
    ema1[lag] = close_1w[:lag+1].mean()
    for i in range(lag+1, len(close_1w)):
        ema1[i] = alpha * close_1w[i] + (1 - alpha) * ema1[i-1]
    
    # Lagged close
    lagged_close = np.roll(close_1w, lag)
    lagged_close[:lag] = np.nan
    
    # dema input: 2*close - lagged_close
    dema_input = 2 * close_1w - lagged_close
    
    # Second EMA of dema_input
    ema2[lag] = dema_input[:lag+1].mean()
    for i in range(lag+1, len(dema_input)):
        if not np.isnan(dema_input[i]):
            ema2[i] = alpha * dema_input[i] + (1 - alpha) * ema2[i-1]
    
    zlema_1w = ema2
    trend_up_1w = close_1w > zlema_1w
    
    # Align trend to 12h
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # TRIX(15,9) on 12h close - momentum oscillator
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) then % change
    period1 = 15
    period2 = 9
    alpha1 = 2.0 / (period1 + 1)
    alpha2 = 2.0 / (period2 + 1)
    
    # Triple EMA
    ema1_trix = np.full(n, np.nan)
    ema2_trix = np.full(n, np.nan)
    ema3_trix = np.full(n, np.nan)
    
    # First EMA
    ema1_trix[period1] = close[:period1+1].mean()
    for i in range(period1+1, n):
        ema1_trix[i] = alpha1 * close[i] + (1 - alpha1) * ema1_trix[i-1]
    
    # Second EMA
    ema2_trix[period1] = ema1_trix[:period1+1].mean()
    for i in range(period1+1, n):
        if not np.isnan(ema1_trix[i]):
            ema2_trix[i] = alpha1 * ema1_trix[i] + (1 - alpha1) * ema2_trix[i-1]
    
    # Third EMA
    ema3_trix[period1] = ema2_trix[:period1+1].mean()
    for i in range(period1+1, n):
        if not np.isnan(ema2_trix[i]):
            ema3_trix[i] = alpha1 * ema2_trix[i] + (1 - alpha1) * ema3_trix[i-1]
    
    # TRIX = % change of triple EMA
    trix = np.full(n, np.nan)
    for i in range(period1+1, n):
        if not np.isnan(ema3_trix[i]) and i > period1+1 and not np.isnan(ema3_trix[i-1]):
            if ema3_trix[i-1] != 0:
                trix[i] = (ema3_trix[i] - ema3_trix[i-1]) / ema3_trix[i-1] * 100
    
    # Signal line: EMA of TRIX
    trix_signal = np.full(n, np.nan)
    trix_signal[period1+period2] = trix[period1+1:period1+period2+2].mean()
    for i in range(period1+period2+1, n):
        if not np.isnan(trix[i]):
            trix_signal[i] = alpha2 * trix[i] + (1 - alpha2) * trix_signal[i-1]
    
    # Volume confirmation: volume > 1.5 * 20-period MA
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            if i > 0:
                vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period1+period2+1, 30)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix_signal[i]) or
            np.isnan(trend_up_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line + uptrend + volume
            if (trix[i] > trix_signal[i] and 
                trix[i-1] <= trix_signal[i-1] and 
                trend_up_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line + downtrend + volume
            elif (trix[i] < trix_signal[i] and 
                  trix[i-1] >= trix_signal[i-1] and 
                  not trend_up_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below signal line or trend changes
            if (trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]) or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above signal line or trend changes
            if (trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]) or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals