#!/usr/bin/env python3
# 6h_KAMA_RSI_Combo_1dTrend
# Hypothesis: Use KAMA trend filter from daily chart to identify regime, then enter long/short based on RSI extremes (below 30/above 70) with volume confirmation on 6h.
# KAMA adapts to market noise, reducing whipsaws in choppy markets. RSI extremes provide mean reversion entries in ranging markets and momentum exhaustion signals in trends.
# Works in bull markets by buying dips in uptrends, in bear markets by selling rallies in downtrends. Volume filter ensures only high-conviction moves trigger entries.
# Target: 20-40 trades/year on 6h timeframe.

name = "6h_KAMA_RSI_Combo_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average ) on daily close
    # Parameters: ER length = 10, Fast SC = 2/(2+1), Slow SC = 2/(30+1)
    er_len = 10
    fast_sc = 2 / (2 + 1)      # 0.6667
    slow_sc = 2 / (30 + 1)     # 0.0645
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.zeros_like(close_1d)
    if len(close_1d) >= er_len:
        for i in range(er_len, len(close_1d)):
            net_change = np.abs(close_1d[i] - close_1d[i-er_len])
            total_change = np.sum(np.abs(np.diff(close_1d[i-er_len:i+1])))
            if total_change > 0:
                er[i] = net_change / total_change
            else:
                er[i] = 0
    
    # Calculate Smoothing Constant (SC)
    sc = np.zeros_like(close_1d)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    if len(close_1d) > 0:
        kama[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 6h close
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[0:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[0:rsi_period])
        for i in range(rsi_period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.zeros_like(close)
    rsi = np.full_like(close, np.nan)
    valid = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss != 0)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi[valid] = 100 - (100 / (1 + rs[valid]))
    
    # Volume filter: 6h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20, 14)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price above KAMA (uptrend) AND RSI oversold (<30) AND volume confirmation
            if close[i] > kama_aligned[i] and rsi[i] < 30 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below KAMA (downtrend) AND RSI overbought (>70) AND volume confirmation
            elif close[i] < kama_aligned[i] and rsi[i] > 70 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought (>70) or price crosses below KAMA (trend change)
            if rsi[i] > 70 or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) or price crosses above KAMA (trend change)
            if rsi[i] < 30 or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals