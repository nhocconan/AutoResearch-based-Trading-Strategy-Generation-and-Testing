#!/usr/bin/env python3
"""
4h_1d_1w_trend_breakout_volume_v1
Hypothesis: Combine 1d trend filter with 1w higher timeframe structure and 4h breakout logic.
Long when 4h price breaks above 4h Donchian high(20) with 1d bullish trend (close>EMA50) and 1w uptrend (close>SMA50).
Short when 4h price breaks below 4h Donchian low(20) with 1d bearish trend (close<EMA50) and 1w downtrend (close<SMA50).
Volume confirmation filters out weak breakouts. Designed for fewer, higher-quality trades in both bull and bear markets.
Target: 20-50 trades/year per symbol (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_trend_breakout_volume_v1"
timeframe = "4h"
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (already aligned but ensure proper handling)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d trend filter: EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1d = close_1d > ema_50_1d
    trend_bearish_1d = close_1d < ema_50_1d
    trend_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # 1w trend filter: SMA(50) for higher timeframe structure
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    trend_uptrend_1w = close_1w > sma_50_1w
    trend_downtrend_1w = close_1w < sma_50_1w
    trend_uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_uptrend_1w.astype(float))
    trend_downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_downtrend_1w.astype(float))
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(trend_bullish_1d_aligned[i]) or np.isnan(trend_bearish_1d_aligned[i]) or
            np.isnan(trend_uptrend_1w_aligned[i]) or np.isnan(trend_downtrend_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or 1d trend turns bearish or 1w trend turns down
            if close[i] < donchian_low_aligned[i] or trend_bearish_1d_aligned[i] > 0.5 or trend_downtrend_1w_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or 1d trend turns bullish or 1w trend turns up
            if close[i] > donchian_high_aligned[i] or trend_bullish_1d_aligned[i] > 0.5 or trend_uptrend_1w_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with 1d bullish trend, 1w uptrend and volume
            if close[i] > donchian_high_aligned[i] and trend_bullish_1d_aligned[i] > 0.5 and trend_uptrend_1w_aligned[i] > 0.5 and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with 1d bearish trend, 1w downtrend and volume
            elif close[i] < donchian_low_aligned[i] and trend_bearish_1d_aligned[i] > 0.5 and trend_downtrend_1w_aligned[i] > 0.5 and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals