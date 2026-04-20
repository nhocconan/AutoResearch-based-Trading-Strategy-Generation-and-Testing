#!/usr/bin/env python3
# 12h_1d_KAMA_Trend_With_Adaptive_Range_Filter
# Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction
# combined with Bollinger Band width percentile to filter for trending/mean-reverting regimes.
# In trending regimes (BBW percentile > 60), follow KAMA direction; in ranging regimes (BBW < 40),
# mean-revert at Bollinger Band edges. Uses daily trend filter from EMA34 to avoid counter-trend trades.
# Designed for low trade frequency (~20-40/year) to minimize fee drag in bear markets.
# Works in bull (follows trend) and bear (mean reversion in ranges) via regime adaptation.

name = "12h_1d_KAMA_Trend_With_Adaptive_Range_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate KAMA (12h) - ER = 10, fast = 2, slow = 30
    lookback = 10
    fast_ema = 2
    slow_ema = 30
    
    change = np.abs(np.diff(close, lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=1) if lookback > 1 else np.zeros_like(change)
    volatility = np.concatenate([np.full(lookback, np.nan), volatility])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[lookback] = close[lookback]
    
    for i in range(lookback + 1, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate Bollinger Band width percentile (20-period) for regime detection
    bb_window = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean()
    std_bb = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std()
    upper_bb = sma_bb + bb_std * std_bb
    lower_bb = sma_bb - bb_std * std_bb
    bb_width = (upper_bb - lower_bb) / sma_bb  # Normalized width
    
    # Percentile of BB width over 50 periods
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback + 1, bb_window + 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        bb_percentile = bb_width_percentile[i]
        is_trending = bb_percentile > 60  # Trending regime
        is_ranging = bb_percentile < 40   # Ranging regime
        
        if position == 0:
            if is_trending:
                # Follow KAMA direction in trending markets
                if close[i] > kama[i] and close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i] and close[i] < ema34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Mean reversion at Bollinger Band edges in ranging markets
                if close[i] < lower_bb[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] > upper_bb[i]:
                    signals[i] = -0.25
                    position = -1
                    
        elif position == 1:
            # Long exit conditions
            if is_trending:
                # Exit trend follow when price crosses below KAMA or trend fails
                if close[i] < kama[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # Exit mean reversion at midline or opposite band
                if close[i] > sma_bb[i] or close[i] >= upper_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:
            # Short exit conditions
            if is_trending:
                # Exit trend follow when price crosses above KAMA or trend fails
                if close[i] > kama[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # Exit mean reversion at midline or opposite band
                if close[i] < sma_bb[i] or close[i] <= lower_bb[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals