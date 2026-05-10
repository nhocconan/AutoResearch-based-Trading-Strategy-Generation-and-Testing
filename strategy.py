#!/usr/bin/env python3
# 1h_4h1d_Confluence_Breakout
# Hypothesis: Use 4h trend and 1d momentum as directional filters, with 1h breakout entries.
# Works in bull/bear: 4h trend filters direction, 1d momentum filters strength, 1h breakout captures entries.
# Target: 15-30 trades/year by requiring confluence of multiple timeframes.

name = "1h_4h1d_Confluence_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mts_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d momentum: RSI14
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h breakout: Donchian20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend, 1d bullish momentum (RSI>50), 1h breaks above Donchian high with volume
            if (trend_4h_up_aligned[i] > 0.5 and
                rsi_1d_aligned[i] > 50 and
                high[i] > donchian_high[i] and
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, 1d bearish momentum (RSI<50), 1h breaks below Donchian low with volume
            elif (trend_4h_down_aligned[i] > 0.5 and
                  rsi_1d_aligned[i] < 50 and
                  low[i] < donchian_low[i] and
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: 4h trend turns down OR price breaks below Donchian low
            if (trend_4h_up_aligned[i] < 0.5 or
                low[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: 4h trend turns up OR price breaks above Donchian high
            if (trend_4h_down_aligned[i] < 0.5 or
                high[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals