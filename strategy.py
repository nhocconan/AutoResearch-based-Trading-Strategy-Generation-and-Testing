#!/usr/bin/env python3
# Hypothesis: 1h mean reversion strategy using 4h Bollinger Band squeeze + 1d trend filter for BTC/ETH.
# In low volatility (BB width < 20th percentile of last 50 4h bars), price tends to revert to 1d EMA50.
# Long when price touches lower BB and close > 1d EMA50; short when price touches upper BB and close < 1d EMA50.
# Uses session filter (08-20 UTC) to reduce noise. Target 15-30 trades/year to minimize fee drag.
# Works in both bull/bear: mean reversion in ranging markets, trend filter avoids counter-trend in strong moves.

name = "1h_BB_Squeeze_MeanRev_4hVol_1dTrend_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # Pre-compute session hours (08-20 UTC) - prices.index is DatetimeIndex
    session_hours = prices.index.hour
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    # 4h Bollinger Bands (20, 2.0)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # BB middle = 20 SMA
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    # BB std = 20 period std dev
    std_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper_4h = sma_20_4h + 2.0 * std_20_4h
    bb_lower_4h = sma_20_4h - 2.0 * std_20_4h
    bb_width_4h = (bb_upper_4h - bb_lower_4h) / sma_20_4h  # normalized width
    
    # Align BB to 1h
    bb_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_upper_4h)
    bb_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_lower_4h)
    bb_width_4h_aligned = align_htf_to_ltf(prices, df_4h, bb_width_4h)
    
    # BB squeeze condition: width < 20th percentile of last 50 bars
    bb_width_series = pd.Series(bb_width_4h_aligned)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=30).quantile(0.20).values
    bb_squeeze = bb_width_4h_aligned < bb_width_percentile
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h Bollinger Bands for entry timing (20, 2.0)
    sma_20_1h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20_1h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper_1h = sma_20_1h + 2.0 * std_20_1h
    bb_lower_1h = sma_20_1h - 2.0 * std_20_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bb_upper_1h[i]) or 
            np.isnan(bb_lower_1h[i]) or 
            np.isnan(bb_squeeze[i]) or
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0  # close position outside session
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches lower 1h BB, in BB squeeze, and close > 1d EMA50 (bullish bias)
            if (low[i] <= bb_lower_1h[i] and 
                bb_squeeze[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price touches upper 1h BB, in BB squeeze, and close < 1d EMA50 (bearish bias)
            elif (high[i] >= bb_upper_1h[i] and 
                  bb_squeeze[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches upper 1h BB or loses squeeze (vol expansion)
            if (high[i] >= bb_upper_1h[i]) or (not bb_squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price touches lower 1h BB or loses squeeze (vol expansion)
            if (low[i] <= bb_lower_1h[i]) or (not bb_squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals