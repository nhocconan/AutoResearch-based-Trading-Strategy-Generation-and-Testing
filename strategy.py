#!/usr/bin/env python3
# 12h_PriceChannel_Breakout_1wTrend_Volume
# Hypothesis: Uses weekly trend + 12h price channel breakout (Donchian 20) + volume confirmation.
# Long when: weekly uptrend (price > 50-week SMA) + 12h price breaks above 20-bar high + volume > 1.5x 20-bar avg.
# Short when: weekly downtrend (price < 50-week SMA) + 12h price breaks below 20-bar low + volume > 1.5x 20-bar avg.
# Exit when: price breaks back through the opposite channel boundary.
# Weekly trend filter avoids whipsaws in ranging markets; breakout captures momentum.
# Works in bull markets by catching early uptrends, in bear by catching early downtrends.
# Volume confirmation reduces false breakouts. Designed for low trade frequency (<30/year).

name = "12h_PriceChannel_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly trend: price vs 50-week SMA ---
    close_1w = df_1w['close'].values
    sma_50 = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        sma_50[i] = np.mean(close_1w[i-50:i])
    weekly_uptrend = close_1w > sma_50
    weekly_downtrend = close_1w < sma_50
    
    # Align weekly trend to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # --- 12h Donchian channel (20-period) ---
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    for i in range(20, n):
        high_max[i] = np.max(high[i-20:i])
        low_min[i] = np.min(low[i-20:i])
    
    # --- Volume confirmation: > 1.5x 20-period average ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of weekly SMA 50, Donchian 20, vol MA 20)
    start_idx = 50  # weekly SMA needs 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_max[i]) or
            np.isnan(low_min[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend from weekly
        is_weekly_up = weekly_uptrend_aligned[i]
        is_weekly_down = weekly_downtrend_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if is_weekly_up and vol_spike:
                # Long: weekly uptrend + volume spike + price breaks above 20-bar high
                if close[i] > high_max[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_weekly_down and vol_spike:
                # Short: weekly downtrend + volume spike + price breaks below 20-bar low
                if close[i] < low_min[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price breaks below 20-bar low (contrarian exit)
                if close[i] < low_min[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above 20-bar high (contrarian exit)
                if close[i] > high_max[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals