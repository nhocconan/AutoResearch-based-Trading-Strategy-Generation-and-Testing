#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_v1
Hypothesis: KAMA trend direction on daily chart + RSI mean reversion + Choppiness regime filter.
- Primary: KAMA(14) on 1d for trend direction (above KAMA = bullish, below = bearish)
- Entry filter: RSI(14) < 30 for long, > 70 for short (mean reversion in trend)
- Regime filter: Choppiness Index(14) > 61.8 for ranging markets (avoid strong trends)
- Volume confirmation: daily volume > 1.5x 20-day average
- Weekly trend filter: price above/below weekly KAMA(34) to avoid counter-trend trades
- Position sizing: 0.25 for long, -0.25 for short
- Target: 15-25 trades/year (60-100 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
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
    
    # Weekly KAMA for trend filter (KAMA with ER=34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio for KAMA
    change_1w = np.abs(np.diff(close_1w, k=34))
    sum_abs_diff_1w = np.zeros_like(close_1w)
    for i in range(34, len(close_1w)):
        sum_abs_diff_1w[i] = np.sum(np.abs(np.diff(close_1w[i-34:i+1])))
    er_1w = np.where(sum_abs_diff_1w != 0, change_1w / sum_abs_diff_1w, 0)
    sc_1w = (er_1w * (2/(34+1) - 2/(2+1)) + 2/(2+1))**2
    kama_1w = np.zeros_like(close_1w)
    kama_1w[34] = close_1w[34]
    for i in range(35, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    
    trend_1w_up = close_1w > kama_1w
    trend_1w_down = close_1w < kama_1w
    
    # Forward fill weekly trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Daily KAMA for trend (KAMA with ER=14)
    change_1d = np.abs(np.diff(close, k=14))
    sum_abs_diff_1d = np.zeros_like(close)
    for i in range(14, len(close)):
        sum_abs_diff_1d[i] = np.sum(np.abs(np.diff(close[i-14:i+1])))
    er_1d = np.where(sum_abs_diff_1d != 0, change_1d / sum_abs_diff_1d, 0)
    sc_1d = (er_1d * (2/(14+1) - 2/(2+1)) + 2/(2+1))**2
    kama_1d = np.zeros_like(close)
    kama_1d[14] = close[14]
    for i in range(15, len(close)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close[i] - kama_1d[i-1])
    
    trend_1d_up = close > kama_1d
    trend_1d_down = close < kama_1d
    
    # RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14) for regime filter
    atr_14 = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14 if i >= 1 else tr[i]
    
    max_high_14 = np.zeros(n)
    min_low_14 = np.zeros(n)
    for i in range(n):
        if i < 14:
            max_high_14[i] = np.max(high[:i+1])
            min_low_14[i] = np.min(low[:i+1])
        else:
            max_high_14[i] = np.max(high[i-14:i+1])
            min_low_14[i] = np.min(low[i-14:i+1])
    
    chop_denom = np.where((max_high_14 - min_low_14) != 0, max_high_14 - min_low_14, 1)
    chop = 100 * np.log10(np.sum(tr) / chop_denom) / np.log10(14)
    chopping = chop > 61.8  # Choppy/ranging market
    
    # Volume filter: daily volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or 
            np.isnan(chopping[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below daily KAMA OR weekly trend turns down OR RSI > 70 (overbought)
            if (close[i] < kama_1d[i]) or trend_1w_down_aligned[i] or (rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price above daily KAMA OR weekly trend turns up OR RSI < 30 (oversold)
            if (close[i] > kama_1d[i]) or trend_1w_up_aligned[i] or (rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price above daily KAMA + weekly uptrend + RSI < 30 + choppy + volume
            if (close[i] > kama_1d[i]) and trend_1w_up_aligned[i] and (rsi[i] < 30) and chopping[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below daily KAMA + weekly downtrend + RSI > 70 + choppy + volume
            elif (close[i] < kama_1d[i]) and trend_1w_down_aligned[i] and (rsi[i] > 70) and chopping[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals