#!/usr/bin/env python3
"""
Experiment #3931: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) capture major trend moves with minimal whipsaws. Volume > 2.0x MA(20) confirms breakout strength. ATR(14) trailing stop (2.5x) manages risk. Target: 75-150 trades over 4 years (19-37/year). Uses discrete sizing (0.25) to minimize fee drag. Works in bull/bear via 1d weekly pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3931_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (calculate from prior week) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot from prior week's daily OHLC (requires 5 days)
    # We'll use the prior week's high, low, close to calculate Camarilla-style weekly pivot
    # For simplicity, we use prior week's range: pivot = (week_high + week_low + week_close) / 3
    # Then R4 = pivot + 1.5 * (week_high - week_low), S4 = pivot - 1.5 * (week_high - week_low)
    # R3 = pivot + 1.0 * (week_high - week_low), S3 = pivot - 1.0 * (week_high - week_low)
    week_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().values  # approx weekly close
    pivot = (week_high + week_low + week_close) / 3.0
    week_range = week_high - week_low
    r4 = pivot + 1.5 * week_range
    s4 = pivot - 1.5 * week_range
    r3 = pivot + 1.0 * week_range
    s3 = pivot - 1.0 * week_range
    
    # Align HTF pivot levels to 6h timeframe (shifted by 1 for completed week only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
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
    
    warmup = max(lookback_dc + 1, 20, 5)  # 5 for weekly calc
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
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
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine bias from weekly pivot levels:
            # Bullish bias: price above R3 (strong) or between S3 and R3 (neutral-bullish)
            # Bearish bias: price below S3 (strong) or between S3 and R3 (neutral-bearish)
            # We'll use: long if price > r3_aligned[i], short if price < s3_aligned[i]
            # But only trade in direction of weekly pivot extreme breakouts for continuation
            # Long: breakout above Donchian upper AND price > r4_aligned (strong bullish)
            # Short: breakdown below Donchian lower AND price < s4_aligned (strong bearish)
            # Mean reversion fade: long if price < s3_aligned AND breaking above Donchian lower?
            # Actually, let's keep it simple: breakout continuation in direction of weekly extreme
            bullish_breakout = price > highest_high[i-1] and price > r4_aligned[i]
            bearish_breakout = price < lowest_low[i-1] and price < s4_aligned[i]
            
            if bullish_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif bearish_breakout:
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