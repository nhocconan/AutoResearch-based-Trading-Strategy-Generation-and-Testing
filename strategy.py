#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based position sizing.
# Uses Donchian channel (20-period high/low) for structure-based breakout entries.
# 1d EMA50 for higher timeframe trend direction filter to avoid counter-trend trades.
# ATR(14) for volatility-adjusted position sizing (inverse vol) and stoploss.
# Discrete position sizing at ±0.25 to minimize fee drag while maintaining edge.
# Target: 80-150 total trades over 4 years (20-38/year) within 4h limits.
# Works in bull markets via breakout continuation and in bear markets via volatility expansion capture.

name = "4h_Donchian20_1dEMA50_ATRVolSizing_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR(14) for volatility and position sizing
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, donchian_period, atr_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_upper = highest_high[i]
        curr_lower = lowest_low[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, above 1d EMA50
            if curr_close > curr_upper and curr_close > curr_ema_50_1d:
                # ATR-based position sizing: inverse volatility
                size = min(0.30, 0.015 / curr_atr * 100)  # scales with volatility
                size = max(0.15, min(size, 0.30))  # clamp to 0.15-0.30
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower band, below 1d EMA50
            elif curr_close < curr_lower and curr_close < curr_ema_50_1d:
                size = min(0.30, 0.015 / curr_atr * 100)
                size = max(0.15, min(size, 0.30))
                signals[i] = -size
                position = -1
        
        elif position == 1:  # Long position
            # ATR trailing stop: exit if price drops 2.0*ATR from entry
            if curr_close < close[i-1] - (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain discrete size
        
        elif position == -1:  # Short position
            # ATR trailing stop: exit if price rises 2.0*ATR from entry
            if curr_close > close[i-1] + (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals