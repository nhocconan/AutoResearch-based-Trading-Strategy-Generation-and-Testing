#!/usr/bin/env python3
# 6h_KeltnerChannel_RSI_Trend_Filter
# Hypothesis: Keltner Channel breakout with RSI momentum filter and trend confirmation.
# Works in bull/bear: Keltner adapts to volatility, RSI avoids overextended entries, trend filter prevents counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size.

name = "6h_KeltnerChannel_RSI_Trend_Filter"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Keltner Channel (20, 2.0) on 6h data
    kc_period = 20
    kc_multiplier = 2.0
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr = np.full_like(close, np.nan)
    if len(tr) >= kc_period:
        atr[kc_period-1] = np.mean(tr[0:kc_period])
        for i in range(kc_period, len(tr)):
            atr[i] = (atr[i-1] * (kc_period-1) + tr[i]) / kc_period
    
    # Keltner Channel
    kc_middle = np.full_like(close, np.nan)
    kc_upper = np.full_like(close, np.nan)
    kc_lower = np.full_like(close, np.nan)
    
    if len(close) >= kc_period:
        # EMA of close for middle line
        ema_close = np.full_like(close, np.nan)
        ema_close[kc_period-1] = np.mean(close[0:kc_period])
        for i in range(kc_period, len(close)):
            ema_close[i] = (ema_close[i-1] * (kc_period-1) + close[i]) / kc_period
        
        kc_middle = ema_close
        kc_upper = kc_middle + (kc_multiplier * atr)
        kc_lower = kc_middle - (kc_multiplier * atr)
    
    # Calculate RSI(14) on 6h data
    rsi_period = 14
    delta = np.diff(close)
    delta = np.concatenate([[0], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[0:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[0:rsi_period])
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.full_like(close, np.nan)
    rsi = np.full_like(close, np.nan)
    valid_loss = avg_loss != 0
    rs[valid_loss] = avg_gain[valid_loss] / avg_loss[valid_loss]
    rsi[valid_loss] = 100 - (100 / (1 + rs[valid_loss]))
    rsi[~valid_loss] = 100  # When no loss, RSI = 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, rsi_period, 50)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper KC AND RSI < 70 (not overbought) AND uptrend (price > EMA50)
            if (close[i] > kc_upper[i] and 
                rsi[i] < 70 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower KC AND RSI > 30 (not oversold) AND downtrend (price < EMA50)
            elif (close[i] < kc_lower[i] and 
                  rsi[i] > 30 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below middle KC OR RSI > 70 (overbought) OR trend reversal (price < EMA50)
            if (close[i] < kc_middle[i] or 
                rsi[i] > 70 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above middle KC OR RSI < 30 (oversold) OR trend reversal (price > EMA50)
            if (close[i] > kc_middle[i] or 
                rsi[i] < 30 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals