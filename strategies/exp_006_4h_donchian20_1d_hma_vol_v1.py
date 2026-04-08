#!/usr/bin/env python3
"""
Experiment #314: 1h Donchian(20) breakout + 4h/1d trend filter + volume confirmation
HYPOTHESIS: Price breaking 1h Donchian(20) channels with 4h HMA(21) and 1d EMA(50) trend alignment captures momentum moves. Volume confirmation (>1.5x) filters false breakouts. Session filter (08-20 UTC) reduces noise. Uses discrete sizing (0.20) to control fee drag. Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe. Works in bull (trend continuation) and bear (mean reversion via volume spikes at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_314_1h_donchian20_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 4h data for HMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HMA(21) on 4h close
    def hma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(series).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(series).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
        return hma_vals.values
    
    hma_4h = hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === HTF: 1d data for EMA trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC (pre-compute hours) ===
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Trend Filters: 4h HMA + 1d EMA alignment ---
        # Uptrend: price above both 4h HMA and 1d EMA
        # Downtrend: price below both 4h HMA and 1d EMA
        uptrend = price > hma_4h_aligned[i] and price > ema_1d_aligned[i]
        downtrend = price < hma_4h_aligned[i] and price < ema_1d_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss (using 2.0*ATR for standard stops) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 20 bars (~20h on 1h)
            if bars_since_entry > 20:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND uptrend
            if breakout_up and uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND downtrend
            elif breakout_down and downtrend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals