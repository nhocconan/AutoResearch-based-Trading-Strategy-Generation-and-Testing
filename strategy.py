# 1d_ChannelBreakout_1wTrend_Volume
# Hypothesis: Price breaks above/below Donchian channel (20) with weekly trend confirmation and volume spike.
# Works in bull markets via breakout follow-through and in bear via mean-reversion off channel extremes.
# Target: 40-80 total trades over 4 years (~10-20/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on weekly close
    close_1w = df_1w['close'].values
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian channel (20) on daily data
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Donchian (20), EMA (50), volume MA (20)
    start_idx = max(lookback, ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below weekly EMA(50)
        uptrend = price > ema_1w_aligned[i]
        downtrend = price < ema_1w_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above upper Donchian band in uptrend with volume
            if price > upper[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below lower Donchian band in downtrend with volume
            elif price < lower[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below lower Donchian band (reversal)
            if price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above upper Donchian band (reversal)
            if price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_ChannelBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0