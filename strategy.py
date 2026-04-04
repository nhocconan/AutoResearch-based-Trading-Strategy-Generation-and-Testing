#!/usr/bin/env python3
"""
Experiment #3252: 12h Donchian Breakout + 1d EMA Trend + Volume Spike + Chop Filter
HYPOTHESIS: 12h Donchian(20) breakouts with 1d EMA(50) trend filter and volume confirmation (>1.5x 20-period average) capture medium-term trends.
Added choppiness regime filter: only trade when CHOP(14) < 61.8 (trending market) to avoid whipsaws in ranging markets.
Position size 0.25. ATR trailing stop (2.0x) manages risk. Designed for low trade frequency ideal for 12h timeframe.
Volume spike threshold lowered from 2.0x to 1.5x to increase trade frequency while maintaining confirmation quality.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3252_12h_donchian20_1d_ema_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 12h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 12h Indicators: Choppiness Index (14-period) for regime filter ===
    def choppiness_index(high, low, close, period=14):
        if len(close) < period:
            return np.full_like(close, np.nan)
        atr_sum = np.zeros(len(close))
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_sum[i] = atr_sum[i-1] + tr
        atr_sum_period = pd.Series(atr_sum).rolling(window=period, min_periods=period).sum().values
        highest_high_period = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low_period = pd.Series(low).rolling(window=period, min_periods=period).min().values
        range_max_min = highest_high_period - lowest_low_period
        chop = 100 * np.log10(atr_sum_period / range_max_min) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20, 14, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Only trade in trending markets (CHOP < 61.8) ---
        if chop[i] >= 61.8:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
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
                # Exit if price rises 2.0*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
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
        # Require volume spike (> 1.5x average) for confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # 1d EMA trend filter: only long above EMA, short below EMA
            price_vs_ema = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above Donchian high with bullish 1d trend
            if price > highest_high[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish 1d trend
            elif price < lowest_low[i] and price_vs_ema < 0:
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