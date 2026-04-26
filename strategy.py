#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyTrend_VolumeConfirmation
Hypothesis: On 6h timeframe, price breaking 20-period Donchian channels in the direction of 1w EMA50 trend with volume confirmation (>1.3x 20-period MA) captures high-probability trend continuation moves. Weekly EMA50 provides robust trend filter resistant to 6h noise, while Donchian breakouts offer clear entry/exit levels. Volume spike confirms institutional participation. Designed for 12-37 trades/year with discrete sizing (±0.25) and ATR-based trailing stop (2.5x) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_series_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 6h ATR(20) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_6h_values = atr_6h.values
    
    # Volume confirmation: volume > 1.3 * 20-period MA on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Donchian (20), ATR (20), volume MA (20) + time for 1w alignment
    start_idx = max(20, 20, 20) + 48  # +48 to ensure 1w bar completion (6h -> 1w: 28 bars per week)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        atr_val = atr_6h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(donch_high) or np.isnan(donch_low) or np.isnan(ema_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > weekly EMA50, bearish when price < weekly EMA50
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Donchian breakout conditions: price breaks channel with trend alignment + volume confirmation
        long_breakout = close_val > donch_high
        short_breakout = close_val < donch_low
        
        long_entry = trend_bullish and long_breakout and vol_conf
        short_entry = trend_bearish and short_breakout and vol_conf
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyTrend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0