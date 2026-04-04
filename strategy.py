#!/usr/bin/env python3
"""
Experiment #4168: 12h Donchian(20) breakout + 1w/1d HTF alignment + volume confirmation
HYPOTHESIS: 12h Donchian breakouts aligned with weekly (1w) and daily (1d) trend direction capture swing trades with reduced whipsaw. Weekly trend (from 1w data) provides structural bias, daily trend (from 1d) confirms intermediate momentum, and volume spike (>1.5x) filters false breakouts. Uses ATR-based trailing stop. Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4168_12h_donchian20_1w1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly trend bias ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 20:
        # Weekly EMA(20) for trend
        weekly_ema = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
        weekly_trend = weekly_ema  # price > EMA = uptrend
    else:
        weekly_trend = np.full(len(df_1w), np.nan)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # === HTF: 1d data for daily trend bias ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        # Daily EMA(20) for trend
        daily_ema = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
        daily_trend = daily_ema  # price > EMA = uptrend
    else:
        daily_trend = np.full(len(df_1d), np.nan)
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # === 12h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 5, 20 + 5, 14 + 5)  # DC lookback, vol MA buffer, EMA buffers, ATR buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_trend_aligned[i]) or np.isnan(daily_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # HTF trend filters: price above EMA = bullish bias, below = bearish bias
            weekly_bullish = price > weekly_trend_aligned[i]
            weekly_bearish = price < weekly_trend_aligned[i]
            daily_bullish = price > daily_trend_aligned[i]
            daily_bearish = price < daily_trend_aligned[i]
            
            # Long conditions: Donchian breakout up + both weekly and daily bullish
            long_entry = breakout_up and weekly_bullish and daily_bullish
            
            # Short conditions: Donchian breakout down + both weekly and daily bearish
            short_entry = breakout_down and weekly_bearish and daily_bearish
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals