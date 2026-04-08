#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_v1
Hypothesis: Daily trend-following using Kaufman Adaptive Moving Average (KAMA) direction
combined with RSI for entry timing and Choppiness Index to filter choppy markets.
KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI avoids overbought/oversold
entries. Choppiness Index > 61.8 indicates ranging markets where we avoid trend trades.
Works in both bull and bear markets by only taking trades when trend is clear and avoiding chop.
Timeframe: 1d, Target: 10-25 trades/year (40-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA Calculation (10-period) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Handle first 9 values where we don't have 10-period data
    change = np.concatenate([np.full(9, np.nan), change])
    vol = np.concatenate([np.full(9, np.nan), vol])
    er = np.where(vol != 0, change / vol, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # === RSI (14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # First average
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (14-period) ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # First TR uses current high-low only
    tr[0] = high[0] - low[0]
    
    atr_sum = np.zeros(n)
    for i in range(14, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])  # 14-period sum
    # Handle first 14 values
    for i in range(1, 14):
        atr_sum[i] = np.sum(tr[0:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i-13)
        highest_high[i] = np.max(high[start_idx:i+1])
        lowest_low[i] = np.min(low[start_idx:i+1])
    
    chop = np.full_like(close, 50.0)  # Default to middle
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    chop = np.where(
        (atr_sum > 0) & (range_hl > 0),
        100 * np.log10(atr_sum / range_hl) / np.log10(14),
        50
    )
    
    # === Weekly Trend Filter (Higher Timeframe) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(20) for trend
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema_20_1w
    trend_1w_down = close_1w < ema_20_1w
    
    # Forward fill weekly trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # === Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below KAMA OR weekly trend turns down OR RSI overbought
            if (close[i] < kama[i]) or trend_1w_down_aligned[i] or (rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price above KAMA OR weekly trend turns up OR RSI oversold
            if (close[i] > kama[i]) or trend_1w_up_aligned[i] or (rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price above KAMA + weekly uptrend + RSI not overbought + low chop + volume
            if (close[i] > kama[i]) and trend_1w_up_aligned[i] and (rsi[i] < 70) and (chop[i] < 61.8) and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price below KAMA + weekly downtrend + RSI not oversold + low chop + volume
            elif (close[i] < kama[i]) and trend_1w_down_aligned[i] and (rsi[i] > 30) and (chop[i] < 61.8) and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals