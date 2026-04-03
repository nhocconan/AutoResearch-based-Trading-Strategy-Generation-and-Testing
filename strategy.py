#!/usr/bin/env python3
"""
Experiment #2094: 1h Donchian(20) breakout + 4h/1d trend filter + volume confirmation
HYPOTHESIS: 1h Donchian breakouts with HTF trend alignment and volume confirmation 
capture institutional order flow with precise entries. Using 4h/1d for signal direction 
and 1h only for entry timing reduces whipsaw. Session filter (08-20 UTC) avoids 
low-liquidity periods. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2094_1h_donchian20_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for Donchian trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian(20) for trend direction
    high_ma_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper_4h = high_ma_4h
    donchian_lower_4h = low_ma_4h
    
    # 4h trend: 1 if close > midpoint, -1 otherwise
    midpoint_4h = (donchian_upper_4h + donchian_lower_4h) / 2
    trend_4h = np.where(close_4h > midpoint_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for Donchian trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Donchian(20) for stronger trend filter
    high_ma_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_1d = high_ma_1d
    donchian_lower_1d = low_ma_1d
    
    # 1d trend: 1 if close > upper, -1 if close < lower, 0 otherwise (neutral)
    trend_1d = np.where(close_1d > donchian_upper_1d, 1, 
                       np.where(close_1d < donchian_lower_1d, -1, 0))
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20) breakout, Volume MA(20), ATR(14) ===
    # 1h Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require BOTH 4h and 1d trend alignment for strong bias filter
        # Only trade when both HTF agree on direction (avoid chop)
        trend_bias_4h = trend_4h_aligned[i]
        trend_bias_1d = trend_1d_aligned[i]
        
        # Skip if HTF trends conflict or are neutral
        if trend_bias_4h == 0 or trend_bias_1d == 0 or trend_bias_4h != trend_bias_1d:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND both HTF up
            if trend_bias_4h > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND both HTF down
            elif trend_bias_4h < 0 and price < donchian_lower[i]:
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