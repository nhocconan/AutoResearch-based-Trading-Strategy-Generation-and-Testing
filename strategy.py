#!/usr/bin/env python3
"""
Experiment #4127: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction capture institutional flow while avoiding counter-trend noise. Weekly pivot (calculated from prior week) provides structural support/resistance that works in both bull and bear markets. Volume confirmation filters false breakouts. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4127_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Weekly pivot points from prior week's OHLC (standard calculation)
        # Pivot = (H + L + C) / 3
        # R1 = 2*P - L, S1 = 2*P - H
        # R2 = P + (H - L), S2 = P - (H - L)
        # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
        # We'll use weekly bias: price above weekly pivot = bullish bias
        
        # Calculate weekly OHLC from daily data (simplified: use last 5 days)
        # In practice, we'd resample to weekly, but using last 5 days as proxy
        week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        week_close = df_1d['close'].values
        
        # Weekly pivot point
        weekly_pivot = (week_high + week_low + week_close) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        
        # Weekly bias: 1 = bullish (price above pivot), -1 = bearish (price below pivot)
        weekly_bias = np.where(week_close > weekly_pivot, 1.0, -1.0)
        weekly_bias_aligned = align_htf_to_ltf(prices, df_1d, weekly_bias)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_bias_aligned = np.zeros(n)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and stoploss ===
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
    
    warmup = max(lookback_dc + 1, 20 + 10, 5 + 10)  # DC lookback, vol MA buffer, weekly buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i])):
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
            
            # Weekly pivot bias filter
            bullish_bias = weekly_bias_aligned[i] > 0
            bearish_bias = weekly_bias_aligned[i] < 0
            
            # Long conditions: Donchian breakout up + bullish weekly bias
            long_entry = breakout_up and bullish_bias
            
            # Short conditions: Donchian breakout down + bearish weekly bias
            short_entry = breakout_down and bearish_bias
            
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