#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA(10) trend + 1d RSI(14) mean reversion + volume filter
# KAMA adapts to market noise, effective in both trending and ranging markets
# 1d RSI < 30 for long, > 70 for short provides mean-reversion edge in extended moves
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Discrete sizing 0.25 targets ~100 total trades over 4 years (25/year)
# Works in bull/bear: KAMA catches trends, RSI avoids exhaustion, volume filters noise

name = "4h_KAMA10_1dRSI14_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA(10) for 4h trend
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[9:], close[:-9]))  # |close[t] - close[t-10]|
    volatility = np.abs(np.subtract(close[1:], close[:-1]))  # |close[t] - close[t-1]|
    
    # Pad volatility for calculation
    volatility_padded = np.concatenate([[np.nan], volatility])
    volatility_sum = pd.Series(volatility_padded).rolling(window=10, min_periods=10).sum().values[9:]
    
    # Calculate ER with padding
    er = np.full_like(close, np.nan)
    er[9:] = np.divide(change, volatility_sum, out=np.zeros_like(change), where=volatility_sum!=0)
    
    # Smoothing constants
    sc = (er * 0.09 + 0.01) ** 2  # where 0.09 = 2/(2+1) for fast EMA, 0.01 = 2/(30+1) for slow EMA
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start with close at period 9
    for i in range(10, len(close)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1d RSI(14)
    delta = np.subtract(close_1d[1:], close_1d[:-1])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + (1.0/period) * (data[i] - result[i-1])
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate volume filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA AND RSI < 30 (oversold) AND volume filter
            if close[i] > kama_aligned[i] and rsi_1d_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND RSI > 70 (overbought) AND volume filter
            elif close[i] < kama_aligned[i] and rsi_1d_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA OR RSI > 70 (overbought)
            if close[i] <= kama_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA OR RSI < 30 (oversold)
            if close[i] >= kama_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals