#!/usr/bin/env python3
"""
Experiment #4114: 1h Donchian(20) breakout + 4h/1d EMA filter + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h/1d EMA trend (EMA50 > EMA200 for bull, EMA50 < EMA200 for bear) and volume confirmation capture institutional order flow. Uses higher timeframes for signal direction, 1h only for entry timing to reduce false breakouts. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4114_1h_donchian20_4h1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h EMA50 and EMA200 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 200:
        close_4h = df_4h['close'].values
        ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
        ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
        # 4h trend: 1 = bull (EMA50 > EMA200), -1 = bear (EMA50 < EMA200), 0 = unclear
        trend_4h = np.where(ema50_4h_aligned > ema200_4h_aligned, 1,
                           np.where(ema50_4h_aligned < ema200_4h_aligned, -1, 0))
    else:
        trend_4h = np.zeros(n)
    
    # === HTF: 1d EMA50 and EMA200 for stronger trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
        ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
        # 1d trend: 1 = bull (EMA50 > EMA200), -1 = bear (EMA50 < EMA200), 0 = unclear
        trend_1d = np.where(ema50_1d_aligned > ema200_1d_aligned, 1,
                           np.where(ema50_1d_aligned < ema200_1d_aligned, -1, 0))
    else:
        trend_1d = np.zeros(n)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(trend_4h[i]) or np.isnan(trend_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
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
            
            # Combined trend filter: require both 4h and 1d to agree
            # Bullish when both trends are bullish (1)
            # Bearish when both trends are bearish (-1)
            bullish_aligned = (trend_4h[i] == 1) and (trend_1d[i] == 1)
            bearish_aligned = (trend_4h[i] == -1) and (trend_1d[i] == -1)
            
            # Long conditions: Donchian breakout up + bullish trend alignment
            long_entry = breakout_up and bullish_aligned
            
            # Short conditions: Donchian breakout down + bearish trend alignment
            short_entry = breakout_down and bearish_aligned
            
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