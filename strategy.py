#!/usr/bin/env python3
# 4h_1d_EMA_Cross_Trend_Filter
# Hypothesis: Long when price > 1d EMA200 (bullish bias) and price breaks above 4h Donchian high with volume confirmation; short when price < 1d EMA200 (bearish bias) and price breaks below 4h Donchian low with volume confirmation. Exit when price crosses back over 1d EMA200. Uses 1d EMA200 as primary trend filter to avoid counter-trend trades, reducing whipsaw in choppy markets. Designed for 20-50 trades/year on 4h timeframe to minimize fee drag.

name = "4h_1d_EMA_Cross_Trend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian channel (20) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.nanmax(high[i-20:i])
        donchian_low[i] = np.nanmin(low[i-20:i])
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1d EMA200 trend
            if close[i] > ema_200_1d_aligned[i]:  # Uptrend bias
                # Long: Breakout above Donchian high with volume confirmation
                if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend bias
                # Short: Breakout below Donchian low with volume confirmation
                if close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below 1d EMA200 (trend change)
            if close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above 1d EMA200 (trend change)
            if close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals