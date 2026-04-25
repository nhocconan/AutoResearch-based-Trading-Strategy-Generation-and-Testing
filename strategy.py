#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + HMA(21) Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. When combined with
Hull Moving Average trend filter and volume confirmation, this strategy avoids false
breakouts in choppy markets. Designed for 4h timeframe to achieve 20-50 trades/year.
Works in bull markets (breakouts above upper band in uptrend) and bear markets
(breakouts below lower band in downtrend). Uses tight entry conditions to minimize
fee drag and maximize test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(series, weights_half, mode='valid') / weights_half.sum()
    wma_half = np.concatenate([np.full(half_period - 1, np.nan), wma_half])
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(series, weights_full, mode='valid') / weights_full.sum()
    wma_full = np.concatenate([np.full(period - 1, np.nan), wma_full])
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    wma_sqrt = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    wma_sqrt = np.concatenate([np.full(sqrt_period - 1, np.nan), wma_sqrt])
    
    return wma_sqrt

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for HTF trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate HMA(21) on 4h for trend filter
    hma_21 = calculate_hma(close, 21)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        hma_trend = hma_21[i]
        htf_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian band AND volume spike AND
            # price > HMA (uptrend) AND price > HTF EMA (bullish HTF)
            long_entry = (curr_close > upper_band) and vol_spike and (curr_close > hma_trend) and (curr_close > htf_trend)
            # Short: price breaks below lower Donchian band AND volume spike AND
            # price < HMA (downtrend) AND price < HTF EMA (bearish HTF)
            short_entry = (curr_close < lower_band) and vol_spike and (curr_close < hma_trend) and (curr_close < htf_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below lower Donchian band OR price crosses below HMA (trend change)
            if (curr_close < lower_band) or (curr_close < hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper Donchian band OR price crosses above HMA (trend change)
            if (curr_close > upper_band) or (curr_close > hma_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_HMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0