#!/usr/bin/env python3
"""
Experiment #4027: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts aligned with 1d weekly pivot levels (R4/S4 for continuation, R3/S3 for fade) 
and volume confirmation capture high-probability moves. Weekly pivots derived from prior 1d weekly bar 
(OHLC) provide institutional reference points. In uptrends (price above weekly pivot), buy upper 
Donchian breakouts; in downtrends (price below weekly pivot), sell lower breakouts. Volume > 1.8x MA20 
filters noise. ATR(20) trailing stop (2.5x) controls drawdown. Discrete sizing (0.25) limits fee churn. 
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4027_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate weekly pivot from prior week's OHLC (using last completed weekly bar)
        # We'll use the 1d data to compute weekly OHLC on the fly
        # For simplicity, we approximate weekly pivot as the average of prior 5-day high, low, close
        # In practice, we'd use actual weekly data, but 1d HTF allows us to compute it
        df_1d_index = pd.RangeIndex(len(df_1d))
        high_1d = pd.Series(df_1d['high'].values, index=df_1d_index)
        low_1d = pd.Series(df_1d['low'].values, index=df_1d_index)
        close_1d = pd.Series(df_1d['close'].values, index=df_1d_index)
        
        # Weekly high/low/close: using last 5 trading days (approximation)
        weekly_high = high_1d.rolling(window=5, min_periods=5).max().shift(1)  # prior week
        weekly_low = low_1d.rolling(window=5, min_periods=5).min().shift(1)
        weekly_close = close_1d.rolling(window=5, min_periods=5).last().shift(1)
        
        # Weekly pivot levels (standard calculation)
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        r3 = weekly_pivot + 2.0 * (weekly_high - weekly_low)
        s3 = weekly_pivot - 2.0 * (weekly_high - weekly_low)
        r4 = weekly_pivot + 3.0 * (weekly_high - weekly_low)
        s4 = weekly_pivot - 3.0 * (weekly_high - weekly_low)
        
        # Align to LTF (6h)
        pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10, 20 + 10, 5 + 5)  # DC lookback, vol MA, ATR buffer, weekly lookback
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
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
        # Require volume spike (> 1.8x average) to filter noise - stricter than 1.5x
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Determine market bias relative to weekly pivot
            price_above_pivot = price > pivot_aligned[i]
            price_below_pivot = price < pivot_aligned[i]
            
            # Breakout logic: 
            # - In uptrend (price > weekly pivot): look for long on upper Donchian breakout
            # - In downtrend (price < weekly pivot): look for short on lower Donchian breakout
            # - Near extreme levels (R3/S3, R4/S4): consider fade signals
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Fade logic at extreme levels
            near_r3 = price > r3_aligned[i] * 0.998  # within 0.2% of R3
            near_s3 = price < s3_aligned[i] * 1.002  # within 0.2% of S3
            
            # Long conditions:
            # 1. Continuation: upper Donchian breakout in uptrend (price > pivot)
            # 2. Fade: price near S3 with bullish rejection (price > low and closing up)
            long_continuation = breakout_up and price_above_pivot
            long_fade = near_s3 and price > low[i] and close[i] > open[i] if hasattr(prices, 'open') else near_s3 and price > lowest_low[i-1]
            
            # Short conditions:
            # 1. Continuation: lower Donchian breakout in downtrend (price < pivot)
            # 2. Fade: price near R3 with bearish rejection (price < high and closing down)
            short_continuation = breakout_down and price_below_pivot
            short_fade = near_r3 and price < high[i] and close[i] < open[i] if hasattr(prices, 'open') else near_r3 and price < highest_high[i-1]
            
            # Prioritize continuation signals (more reliable)
            if long_continuation:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_continuation:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            elif long_fade and not short_fade:  # avoid conflicting signals
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE * 0.5  # smaller size for fade
            elif short_fade and not long_fade:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.5  # smaller size for fade
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals