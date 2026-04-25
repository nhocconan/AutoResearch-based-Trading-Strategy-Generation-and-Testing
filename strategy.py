#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dTrendFilter_v3
Hypothesis: Trade 1h Camarilla R1/S1 breakouts aligned with 4h and 1d trend during UTC 08-20 session. Uses volume confirmation (vol_ratio > 2.2) and discrete sizing (0.20) to target 15-35 trades/year. Added stricter volume threshold (2.2) and removed redundant 1h EMA50 filter to reduce overtrading and improve robustness in both bull and bear markets.
"""

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
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h and 1d data for HTF filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA34 for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1h ATR(14) for Camarilla levels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h Camarilla levels (based on previous day's range)
    lookback = 24
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    closest_close = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
        closest_close[i] = close[i-lookback]
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    for i in range(lookback-1, n):
        if not (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(closest_close[i])):
            range_val = highest_high[i] - lowest_low[i]
            R1[i] = closest_close[i] + range_val * 1.1 / 12
            S1[i] = closest_close[i] - range_val * 1.1 / 12
    
    # Calculate 1h volume ratio (current vs 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(lookback, 50, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if outside session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h and 1d trend
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)[i]
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(close_4h_aligned) or np.isnan(close_1d_aligned):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
            
        htf_4h_bullish = close_4h_aligned > ema_50_4h_aligned[i]
        htf_1d_bullish = close_1d_aligned > ema_34_1d_aligned[i]
        
        # Volume confirmation: need significant spike (increased to 2.2 to reduce overtrading)
        volume_confirmed = vol_ratio[i] > 2.2
        
        if position == 0:
            # Long setup: price breaks above R1 + 4h bullish trend + 1d bullish trend + volume
            long_setup = (close[i] > R1[i]) and htf_4h_bullish and htf_1d_bullish and volume_confirmed
            
            # Short setup: price breaks below S1 + 4h bearish trend + 1d bearish trend + volume
            short_setup = (close[i] < S1[i]) and (not htf_4h_bullish) and (not htf_1d_bullish) and volume_confirmed
            
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
            # Exit: price breaks below S1 OR 1d trend turns bearish
            if (close[i] < S1[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price breaks above R1 OR 1d trend turns bullish
            if (close[i] > R1[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dTrendFilter_v3"
timeframe = "1h"
leverage = 1.0