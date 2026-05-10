#!/usr/bin/env python3
# 12h_1d_RSI_MeanReversion_with_Volume_Filter
# Hypothesis: On 12h timeframe, use RSI(14) from daily timeframe for mean reversion signals
# (RSI < 30 for long, RSI > 70 for short) combined with volume confirmation (>1.5x average)
# and 1-week trend filter (price above/below EMA50) to avoid counter-trend trades.
# Designed to work in both bull and bear markets by fading extremes with trend alignment.
# Target: 20-40 trades/year to minimize fee drag on 12h timeframe.

name = "12h_1d_RSI_MeanReversion_with_Volume_Filter"
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
    
    # 1d RSI(14) for mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan
    
    # RSI levels for mean reversion
    rsi_oversold = rsi_1d < 30
    rsi_overbought = rsi_1d > 70
    
    # Align RSI signals to 12h
    rsi_oversold_aligned = align_htf_to_ltf(prices, df_1d, rsi_oversold.astype(float))
    rsi_overbought_aligned = align_htf_to_ltf(prices, df_1d, rsi_overbought.astype(float))
    
    # 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma[i] = vol_sum / 24
        else:
            vol_ma[i] = np.nan
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_oversold_aligned[i]) or np.isnan(rsi_overbought_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold on 1d + volume confirmation + 1w uptrend
            if (rsi_oversold_aligned[i] > 0.5 and
                volume_confirm[i] and
                trend_1w_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought on 1d + volume confirmation + 1w downtrend
            elif (rsi_overbought_aligned[i] > 0.5 and
                  volume_confirm[i] and
                  trend_1w_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (50) or 1w trend turns down
            if (rsi_1d[-1] >= 50 if len(rsi_1d) > 0 else False) or \
               (i < len(rsi_oversold_aligned) and rsi_oversold_aligned[i] < 0.5) or \
               trend_1w_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (50) or 1w trend turns up
            if (rsi_1d[-1] <= 50 if len(rsi_1d) > 0 else False) or \
               (i < len(rsi_overbought_aligned) and rsi_overbought_aligned[i] < 0.5) or \
               trend_1w_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals