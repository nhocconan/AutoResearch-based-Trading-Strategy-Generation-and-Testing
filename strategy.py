#!/usr/bin/env python3
"""
1h_HTF_Trend_LT_Reversal_v1
Hypothesis: In 1h timeframe, trade mean reversions (pullbacks) aligned with 4h and 1d trend during UTC 08-20 session.
Enter long when price touches 1h EMA20 during 4h/1d uptrend with volume confirmation; enter short when price touches 1h EMA20 during 4h/1d downtrend with volume confirmation.
Uses discrete sizing (0.20) and session filter to target ~20-40 trades/year. Designed to work in both bull (buy pullbacks) and bear (sell rallies) markets.
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h and 1d data for HTF trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA50 for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h EMA20 for entry timing (mean reversion target)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1h ATR(14) for stop/reference (not used in entry, but for context)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h volume ratio (current vs 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(50, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h and 1d trend
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)[i]
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(close_4h_aligned) or np.isnan(close_1d_aligned):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        htf_4h_bullish = close_4h_aligned > ema_50_4h_aligned[i]
        htf_1d_bullish = close_1d_aligned > ema_50_1d_aligned[i]
        htf_bullish = htf_4h_bullish and htf_1d_bullish  # Both timeframes bullish
        htf_bearish = (not htf_4h_bullish) and (not htf_1d_bullish)  # Both timeframes bearish
        
        # Volume confirmation: need significant spike
        volume_confirmed = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long setup: price touches EMA20 from below (pullback in uptrend) + HTF bullish + volume
            long_setup = (close[i] <= ema_20[i] * 1.001) and (close[i] >= ema_20[i] * 0.999) and htf_bullish and volume_confirmed
            
            # Short setup: price touches EMA20 from above (rally in downtrend) + HTF bearish + volume
            short_setup = (close[i] <= ema_20[i] * 1.001) and (close[i] >= ema_20[i] * 0.999) and htf_bearish and volume_confirmed
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price moves above EMA20 (pullback complete) OR HTF trend turns bearish
            if (close[i] > ema_20[i] * 1.005) or (not htf_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price moves below EMA20 (rally complete) OR HTF trend turns bullish
            if (close[i] < ema_20[i] * 0.995) or (htf_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_HTF_Trend_LT_Reversal_v1"
timeframe = "1h"
leverage = 1.0