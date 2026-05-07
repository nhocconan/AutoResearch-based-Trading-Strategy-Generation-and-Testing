#!/usr/bin/env python3
name = "1d_1w_Donchian20_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(10) for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian(20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with weekly uptrend and volume
            if (close[i] > donchian_high[i] and close[i] > ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with weekly downtrend and volume
            elif (close[i] < donchian_low[i] and close[i] < ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below lower Donchian or trend change
            if close[i] < donchian_low[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above upper Donchian or trend change
            if close[i] > donchian_high[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian(20) breakout with weekly EMA(10) trend filter and volume confirmation.
# In weekly uptrends, buy breakouts above daily Donchian high; in weekly downtrends, sell breakdowns below daily Donchian low.
# Exit when price returns to the opposite Donchian band or weekly trend changes.
# Volume filter ensures trades occur with participation. Position size 0.25 controls risk.
# Works in both bull (trend following) and bear (mean reversion at extremes) markets.