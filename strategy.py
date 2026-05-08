#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w = (close_1w > ema20_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Weekly volume spike: current volume > 2.0 * 10-week average
    volume_1w = df_1w['volume'].values
    vol_ma10w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    vol_spike_1w = volume_1w > (vol_ma10w * 2.0)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # 6h Donchian(20) breakout levels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback - 1)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with weekly uptrend and volume spike
            long_cond = (close[i] > highest_high[i] and trend_1w_aligned[i] > 0.5 and vol_spike_1w_aligned[i])
            
            # Short entry: price breaks below Donchian low with weekly downtrend and volume spike
            short_cond = (close[i] < lowest_low[i] and trend_1w_aligned[i] < 0.5 and vol_spike_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low (mean reversion)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high (mean reversion)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume spike.
# Weekly EMA20 ensures alignment with longer-term trend, reducing counter-trend trades.
# Weekly volume spike (2.0x 10-week average) confirms institutional participation.
# Exit on opposite Donchian break for mean reversion in ranging markets.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite level).
# Target: 50-150 total trades over 4 years to minimize fee decay while capturing significant moves.