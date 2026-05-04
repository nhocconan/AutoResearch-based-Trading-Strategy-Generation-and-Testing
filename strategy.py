#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. 1w EMA50 ensures alignment with
# the major trend to avoid counter-trend trades. Volume confirmation (1.8x 20-period EMA)
# filters weak breakouts. Designed for 12h timeframe to target 12-37 trades/year (50-150 total)
# with discrete sizing (0.25). Works in bull markets by buying breakouts in uptrends and in
# bear markets by selling breakdowns in downtrends, avoiding range-bound whipsaws.

name = "12h_Donchian20_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels using prior 20 periods (lookback)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            continue
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 1.8x 20-period EMA on volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + price above 1w EMA50 (uptrend)
            if (close[i] > donchian_h[i] and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + price below 1w EMA50 (downtrend)
            elif (close[i] < donchian_l[i] and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low OR price below 1w EMA50 (trend change)
            if close[i] < donchian_l[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian high OR price above 1w EMA50 (trend change)
            if close[i] > donchian_h[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals