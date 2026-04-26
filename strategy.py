#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_WeeklyTrendFilter
Hypothesis: Weekly Donchian channel breakout with weekly EMA trend filter on daily timeframe.
Enters long when price breaks above weekly Donchian(20) high and weekly EMA50 is rising;
enters short when price breaks below weekly Donchian(20) low and weekly EMA50 is falling.
Uses ATR-based stoploss and discrete sizing 0.25 to limit trades (~10-20/year).
Works in bull/bear via weekly trend filter and price channel structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for HTF trend and Donchian
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donchian_high = high_1w_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1w_series.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate ATR for volatility filtering and stoploss (daily)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 50 for weekly EMA, 20 for Donchian, 14 for ATR
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        atr_val = atr[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long entry: price breaks above weekly Donchian high AND weekly EMA50 is rising
            # Short entry: price breaks below weekly Donchian low AND weekly EMA50 is falling
            long_entry = close_val > donchian_high_val and ema_50_val > ema_50_1w_aligned[i-1]
            short_entry = close_val < donchian_low_val and ema_50_val < ema_50_1w_aligned[i-1]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or ATR-based stop
            # Exit if weekly EMA50 turns bearish OR price drops 2*ATR from entry
            if ema_50_val < ema_50_1w_aligned[i-1] or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or ATR-based stop
            # Exit if weekly EMA50 turns bullish OR price rises 2*ATR from entry
            if ema_50_val > ema_50_1w_aligned[i-1] or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchianBreakout_WeeklyTrendFilter"
timeframe = "1d"
leverage = 1.0