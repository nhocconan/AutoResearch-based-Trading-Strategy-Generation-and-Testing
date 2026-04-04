#!/usr/bin/env python3
"""
Experiment #3487: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1w HTF) 
and volume confirmation capture structural breaks with controlled frequency. 
Weekly pivot provides longer-term bias, Donchian captures breakouts, volume confirms strength.
Works in bull (breakouts with weekly bias) and bear (breakdowns with weekly bias).
Target: 75-150 total trades over 4 years (19-37/year). Position size 0.25.
Uses 1w for pivot direction, 6h only for entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3487_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    # For each 1d bar, we need the prior week's (Monday to Friday) OHLC
    # We'll approximate using rolling window of 5 days for simplicity
    lookback_week = 5
    weekly_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().values
    weekly_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().values
    weekly_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).mean().values  # approx weekly close
    
    # Calculate pivot point and support/resistance levels
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    r2 = pivot_point + (weekly_high - weekly_low)
    s2 = pivot_point - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot_point - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot_point)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r2 + (weekly_high - weekly_low))  # R4 = R3 + range
    s4_aligned = align_htf_to_ltf(prices, df_1d, s2 - (weekly_high - weekly_low))  # S4 = S3 - range
    
    # === 6h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_6h = 20
    highest_high_6h = pd.Series(high).rolling(window=lookback_6h, min_periods=lookback_6h).max().values
    lowest_low_6h = pd.Series(low).rolling(window=lookback_6h, min_periods=lookback_6h).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_6h, lookback_week, 20, 14)
    
    for i in range(warmup, n):
        # --- Session Filter ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 6h Donchian channel (mean reversion)
                elif price <= highest_high_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters 6h Donchian channel (mean reversion)
                elif price >= lowest_low_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Weekly pivot direction filter: 
            # Long bias: price above R3 (bullish weekly structure)
            # Short bias: price below S3 (bearish weekly structure)
            price_vs_r3 = price - r3_aligned[i]
            price_vs_s3 = price - s3_aligned[i]
            
            # Long entry: price breaks above 6h Donchian high with weekly bullish bias (above R3)
            if (price > highest_high_6h[i] and 
                price_vs_r3 > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian low with weekly bearish bias (below S3)
            elif (price < lowest_low_6h[i] and 
                  price_vs_s3 < 0):
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