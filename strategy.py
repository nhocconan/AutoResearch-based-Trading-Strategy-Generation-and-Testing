#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d trend filter
# - Uses 6h Donchian channel breakout (20-period) for entry signals
# - Confirms with 12h volume > 1.8x its 30-period average (strong participation)
# - Uses 1d EMA(50) as trend filter: long only when close > EMA50, short only when close < EMA50
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Donchian breakouts capture trends, volume filter reduces false breakouts, EMA filter avoids counter-trend trades

name = "6h_12h_1d_donchian_volume_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 40 or len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    volume_12h = df_12h['volume'].values
    
    # 12h Volume > 1.8x 30-period average (stricter for fewer trades)
    avg_volume_30 = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    volume_spike_12h = volume_12h > (1.8 * avg_volume_30)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_spike_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for breakout entries
            # Long breakout: price breaks above Donchian high with volume confirmation and uptrend
            if (high[i] >= highest_high[i] and 
                volume_spike_12h_aligned[i] and 
                uptrend):
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks below Donchian low with volume confirmation and downtrend
            elif (low[i] <= lowest_low[i] and 
                  volume_spike_12h_aligned[i] and 
                  downtrend):
                position = -1
                signals[i] = -0.25
        elif position == 1:  # Long position - exit on trend reversal or opposite breakout
            # Exit: trend turns down OR price breaks below Donchian low (failed breakout)
            if (not uptrend or low[i] <= lowest_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position - exit on trend reversal or opposite breakout
            # Exit: trend turns up OR price breaks above Donchian high (failed breakout)
            if (not downtrend or high[i] >= highest_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals