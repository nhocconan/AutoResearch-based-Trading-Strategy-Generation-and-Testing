#!/usr/bin/env python3
"""
Experiment #2969: 4h Donchian Breakout + 1d/1w HTF Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture medium-term trends with controlled frequency.
HTF trend from 1d EMA50 vs EMA200 provides directional bias: only take longs when 1d EMA50 > EMA200, shorts when EMA50 < EMA200.
Weekly trend from 1w close > open adds additional filter for strong momentum alignment.
Volume spike (>1.8x 20-period average) confirms breakout strength. This combination filters false breakouts
while maintaining sufficient trade frequency (target: 75-200 total trades over 4 years) for statistical validity.
ATR-based trailing stop (2.5x) manages risk. Position size 0.25 balances return and drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2969_4h_donchian20_1d_1w_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMAs for trend bias
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    # 1d trend: bullish when EMA50 > EMA200
    trend_1d_bullish = ema50_1d > ema200_1d
    
    # Align 1d trend to 4h timeframe (shifted by 1 for completed bars only)
    trend_1d_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_bullish.astype(np.float64))
    
    # === HTF: 1w data for weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    # Weekly trend: bullish when weekly close > open
    trend_1w_bullish = close_1w > open_1w
    
    # Align 1w trend to 4h timeframe (shifted by 1 for completed bars only)
    trend_1w_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_bullish.astype(np.float64))
    
    # === 4h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])
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
    
    warmup = max(50, lookback, 20, 200)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1d_bullish_aligned[i]) or np.isnan(trend_1w_bullish_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Get HTF trend biases
            trend_1d_bull = bool(trend_1d_bullish_aligned[i] > 0.5)
            trend_1w_bull = bool(trend_1w_bullish_aligned[i] > 0.5)
            
            # Long entry: price breaks above Donchian high with bullish HTF bias
            if price > highest_high[i] and trend_1d_bull and trend_1w_bull:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish HTF bias
            elif price < lowest_low[i] and (not trend_1d_bull) and (not trend_1w_bull):
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