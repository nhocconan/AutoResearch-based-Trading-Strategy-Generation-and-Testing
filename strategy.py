#!/usr/bin/env python3
# 12h_Donchian20_1wTrend_Volume
# Hypothesis: Uses weekly trend (1w EMA) to determine direction, 12h Donchian breakout for entry, and volume confirmation.
# Weekly trend filter avoids counter-trend trades in choppy markets. Donchian breakout captures momentum.
# Volume confirmation ensures institutional participation. Target: 15-25 trades/year per symbol.
# Works in bull markets via breakouts and in bear markets via shorting breakdowns with weekly trend alignment.

timeframe = "12h"
name = "12h_Donchian20_1wTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Donchian channel (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x average volume (24-period = 12 days on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20, 24)  # Ensure we have EMA40, Donchian, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume, and weekly trend is bullish (close > weekly EMA40)
            if (high[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume, and weekly trend is bearish (close < weekly EMA40)
            elif (low[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low (mean reversion or trend exhaustion)
            if low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high (mean reversion or trend exhaustion)
            if high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals