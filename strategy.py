#!/usr/bin/env python3
"""
Experiment #5019: 6h Donchian(20) Breakout + 12h ATR Trend Filter + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts in direction of 12h ATR-based trend (price > SMA50 + ATR*0.5 for long, price < SMA50 - ATR*0.5 for short) with volume confirmation (>1.5x average) capture strong momentum moves while avoiding whipsaws. Uses ATR(14) trailing stop (2.0x) to limit downside. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend) by using adaptive trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5019_6h_donchian20_12h_atr_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for ATR-based trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: SMA50 and ATR14 for trend filter ===
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        
        # SMA50
        sma_50 = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
        
        # ATR14
        tr1 = high_12h[1:] - low_12h[1:]
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Trend filter: long when price > SMA50 + 0.5*ATR, short when price < SMA50 - 0.5*ATR
        trend_long = close_12h > (sma_50 + 0.5 * atr_12h)
        trend_short = close_12h < (sma_50 - 0.5 * atr_12h)
    else:
        sma_50 = np.full(len(df_12h), np.nan)
        atr_12h = np.full(len(df_12h), np.nan)
        trend_long = np.zeros(len(df_12h), dtype=bool)
        trend_short = np.zeros(len(df_12h), dtype=bool)
    
    # Align HTF trend filters to 6h timeframe
    if len(df_12h) > 0:
        trend_long_aligned = align_htf_to_ltf(prices, df_12h, trend_long.astype(np.float64))
        trend_short_aligned = align_htf_to_ltf(prices, df_12h, trend_short.astype(np.float64))
        atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    else:
        trend_long_aligned = np.full(n, np.nan)
        trend_short_aligned = np.full(n, np.nan)
        atr_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 50, 14)  # Donchian, Volume MA, SMA50, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(trend_long_aligned[i]) or np.isnan(trend_short_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with ATR-based trend alignment
        breakout_long = (price >= high_roll[i]) and trend_long_aligned[i] > 0.5 and vol_confirm
        breakout_short = (price <= low_roll[i]) and trend_short_aligned[i] > 0.5 and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals