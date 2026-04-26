#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum and Choppiness Index(14) for regime filtering.
Long when KAMA slopes up, RSI > 50, and CHOP < 61.8 (trending regime).
Short when KAMA slopes down, RSI < 50, and CHOP < 61.8.
Uses weekly EMA200 as higher-timeframe trend filter to avoid counter-trend trades.
Designed for 30-100 total trades over 4 years (7-25/year) with discrete sizing (0.25) to minimize fee drag.
Works in bull markets via trend alignment and in bear markets via short side + regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_sc = 0.666  # 2/(2+1)
    slow_sc = 0.0645 # 2/(30+1)
    
    # Calculate Efficiency Ratio and SMA for KAMA
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation: |net change| / sum(|abs change|) over lookback
    er_lookback = 10
    net_change = np.abs(np.subtract(close[er_lookback:], close[:-er_lookback]))
    sum_abs_change = np.zeros_like(close)
    for i in range(er_lookback, len(close)):
        sum_abs_change[i] = np.sum(np.abs(np.diff(close[i-er_lookback:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = sum_abs_change != 0
    er[mask] = net_change[mask-er_lookback+1] / sum_abs_change[mask] if er_lookback > 0 else 0
    er = np.where(np.isnan(er), 0, er)
    
    # Smoothing constant
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_lookback] = close[er_lookback]  # seed
    for i in range(er_lookback + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # default to neutral
    
    # Choppiness Index(14)
    chop_period = 14
    atr_temp = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_temp[0] = high[0] - low[0]  # first period
    tr_sum = pd.Series(atr_temp).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = np.zeros_like(close)
    mask = (highest_high - lowest_low) != 0
    chop[mask] = 100 * np.log10(tr_sum[mask] / (highest_high[mask] - lowest_low[mask])) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50, chop)  # default to middle
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of KAMA lookback, RSI period, Chop period, WMA period
    start_idx = max(er_lookback + 1, rsi_period, chop_period, 200) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        close_val = close[i]
        ema_200_val = ema_200_1w_aligned[i]
        
        # KAMA slope: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, CHOP < 61.8 (trending), price above weekly EMA200
            long_signal = kama_rising and (rsi_val > 50) and (chop_val < 61.8) and (close_val > ema_200_val)
            
            # Short: KAMA falling, RSI < 50, CHOP < 61.8 (trending), price below weekly EMA200
            short_signal = kama_falling and (rsi_val < 50) and (chop_val < 61.8) and (close_val < ema_200_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA falling OR RSI < 40 OR CHOP > 61.8 (choppy) OR price below weekly EMA200
            if (kama_falling) or (rsi_val < 40) or (chop_val > 61.8) or (close_val < ema_200_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA rising OR RSI > 60 OR CHOP > 61.8 (choppy) OR price above weekly EMA200
            if (kama_rising) or (rsi_val > 60) or (chop_val > 61.8) or (close_val > ema_200_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0