#!/usr/bin/env python3
"""
Experiment #3071: 6h Donchian(20) Breakout + 1d Weekly Pivot + Volume Spike
HYPOTHESIS: 6h Donchian breakouts capture medium-term trends with controlled frequency. 
Weekly pivot levels (from 1d data: P, R1-4, S1-4) act as dynamic support/resistance: 
breakouts above R4 or below S4 indicate strong momentum; reversals at R3/S3 offer mean-reversion entries. 
Volume spike (>2.0x 20-period average) confirms participation. ATR trailing stop (2.5x) manages risk. 
Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year). Designed for both bull 
(trend continuation from pivots) and bear (mean reversion at pivot levels) markets by using price 
channels and volatility filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3071_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    # For each 6h bar, use the most recent completed weekly pivot
    # Weekly pivot: P = (H_prev_week + L_prev_week + C_prev_week) / 3
    # R1 = 2*P - L_prev_week, S1 = 2*P - H_prev_week
    # R2 = P + (H_prev_week - L_prev_week), S2 = P - (H_prev_week - L_prev_week)
    # R3 = H_prev_week + 2*(P - L_prev_week), S3 = L_prev_week - 2*(H_prev_week - P)
    # R4 = R3 + (H_prev_week - L_prev_week), S4 = S3 - (H_prev_week - L_prev_week)
    
    # We need to group by week. Since we don't have direct week grouping,
    # we approximate: use prior 5 daily bars (1 trading week) for pivot calculation
    lookback_week = 5
    if len(high_1d) >= lookback_week:
        # Rolling window of prior 5 days (excluding current)
        high_week = pd.Series(high_1d).shift(1).rolling(window=lookback_week, min_periods=lookback_week).max().values
        low_week = pd.Series(low_1d).shift(1).rolling(window=lookback_week, min_periods=lookback_week).min().values
        close_week = pd.Series(close_1d).shift(1).rolling(window=lookback_week, min_periods=lookback_week).last().values
        
        # Weekly pivot calculation
        P = (high_week + low_week + close_week) / 3.0
        R1 = 2 * P - low_week
        S1 = 2 * P - high_week
        R2 = P + (high_week - low_week)
        S2 = P - (high_week - low_week)
        R3 = high_week + 2 * (P - low_week)
        S3 = low_week - 2 * (high_week - P)
        R4 = R3 + (high_week - low_week)
        S4 = S3 - (high_week - low_week)
        
        # Align to 6h timeframe
        P_aligned = align_htf_to_ltf(prices, df_1d, P)
        R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
        S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
        R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
        S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
        R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
        S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
        R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
        S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    else:
        # Not enough data, fill with NaN
        P_aligned = R1_aligned = S1_aligned = R2_aligned = S2_aligned = np.full(n, np.nan)
        R3_aligned = S3_aligned = R4_aligned = S4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
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
    
    warmup = max(50, lookback, 20, 14, lookback_week)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(P_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i])):
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
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
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
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
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
            # Long entry: price breaks above Donchian high AND above R4 (strong bullish breakout)
            # OR price reverses from S3 with bullish momentum (mean reversion)
            bullish_breakout = price > highest_high[i] and price > R4_aligned[i]
            bullish_reversal = price > S3_aligned[i] and price < lowest_low[i] * 1.001  # near low, reversing up
            
            # Short entry: price breaks below Donchian low AND below S4 (strong bearish breakdown)
            # OR price reverses from R3 with bearish momentum (mean reversion)
            bearish_breakout = price < lowest_low[i] and price < S4_aligned[i]
            bearish_reversal = price < R3_aligned[i] and price > highest_high[i] * 0.999  # near high, reversing down
            
            if bullish_breakout or bullish_reversal:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif bearish_breakout or bearish_reversal:
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