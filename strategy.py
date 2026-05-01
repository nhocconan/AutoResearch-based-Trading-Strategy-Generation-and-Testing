#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
# Uses 1d EMA50 for robust long-term trend alignment that works in both bull and bear markets.
# Long when price breaks above Donchian upper band AND price > 1d EMA50.
# Short when price breaks below Donchian lower band AND price < 1d EMA50.
# Exit when price crosses opposite Donchian band or trend changes (price crosses 1d EMA50).
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Donchian20_1dEMA50_Trend_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1d trend: price above/below EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # warmup for Donchian channels
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Donchian breakout signals
        breakout_up = curr_high > upper_band[i]  # break above upper band
        breakout_down = curr_low < lower_band[i]  # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper band AND price > 1d EMA50
            if breakout_up and price_above_ema[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band AND price < 1d EMA50
            elif breakout_down and price_below_ema[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower band (stoploss) OR price < 1d EMA50 (trend change)
            if curr_low < lower_band[i] or not price_above_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper band (stoploss) OR price > 1d EMA50 (trend change)
            if curr_high > upper_band[i] or not price_below_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals