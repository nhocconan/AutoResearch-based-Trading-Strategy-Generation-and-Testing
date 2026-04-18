#!/usr/bin/env python3
"""
1h_1hr_4hTrend_1dVolatilityBreakout
Hypothesis: In low-volatility regimes (daily volatility < 20th percentile), breakouts from the prior 4h candle's high/low capture momentum with higher reliability. 
Trades only during active session (08-20 UTC) to avoid low-liquidity periods. 
Position size fixed at 0.20 to manage drawdown and reduce turnover. 
Uses 4h trend (close > open) as directional filter and daily volatility percentile as regime filter.
Target: ~20-40 trades/year per symbol to stay under fee drag threshold.
"""

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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h trend direction: bullish if close > open, bearish if close < open
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    bullish_4h = df_4h['close'] > df_4h['open']
    bearish_4h = df_4h['close'] < df_4h['open']
    trend_4h_bullish = align_htf_to_ltf(prices, df_4h, bullish_4h.values)
    trend_4h_bearish = align_htf_to_ltf(prices, df_4h, bearish_4h.values)
    
    # Daily volatility (ATR-based) and percentile rank
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Percentile rank of current ATR over 50-day lookback
    def percentile_rank(arr, window=50):
        rank = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if i < window:
                continue
            window_data = arr[i-window:i]
            if np.all(np.isnan(window_data)):
                rank[i] = np.nan
            else:
                valid = window_data[~np.isnan(window_data)]
                if len(valid) == 0:
                    rank[i] = np.nan
                else:
                    rank[i] = np.sum(valid <= arr[i]) / len(valid) * 100
        return rank
    
    atr_percentile = percentile_rank(atr_1d, window=50)
    low_vol_regime = atr_percentile < 20  # Bottom 20% = low volatility
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    
    # Prior 4h candle high/low for breakout levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    breakout_high = align_htf_to_ltf(prices, df_4h, high_4h)
    breakout_low = align_htf_to_ltf(prices, df_4h, low_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for ATR percentile
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(breakout_high[i]) or np.isnan(breakout_low[i]) or 
            np.isnan(low_vol_aligned[i]) or np.isnan(trend_4h_bullish[i]) or 
            np.isnan(trend_4h_bearish[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = low_vol_aligned[i]
        bullish = trend_4h_bullish[i]
        bearish = trend_4h_bearish[i]
        
        if position == 0:
            # Long: bullish 4h trend + low vol regime + break above prior 4h high
            if bullish and vol_ok and price > breakout_high[i]:
                signals[i] = 0.20
                position = 1
            # Short: bearish 4h trend + low vol regime + break below prior 4h low
            elif bearish and vol_ok and price < breakout_low[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long if 4h trend turns bearish or price breaks below prior 4h low
            if not bullish or price < breakout_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short if 4h trend turns bullish or price breaks above prior 4h high
            if not bearish or price > breakout_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_1hr_4hTrend_1dVolatilityBreakout"
timeframe = "1h"
leverage = 1.0