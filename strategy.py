#!/usr/bin/env python3
"""
Experiment #568: 12h Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Donchian breakouts on 12h timeframe aligned with weekly HMA trend (bullish when price > HMA, bearish when price < HMA) and volume confirmation (>1.5x average) capture institutional flow with minimal overtrading. Weekly HMA provides robust trend filter that works in both bull and bear markets by identifying the dominant trend direction. Volume confirmation ensures participation and reduces false breakouts. Targets 50-150 total trades over 4 years by using tight entry conditions (breakout + HMA alignment + volume) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_568_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on weekly close
    def hma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan)
        half = int(period / 2)
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(series).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(series).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma2 - wma1
        hma_values = pd.Series(raw_hma).ewm(span=sqrt, adjust=False).mean()
        return hma_values.values
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 12h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian and HMA calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- HMA Trend Alignment ---
        # Bullish alignment: price above weekly HMA
        bullish_alignment = price > hma_21_1w_aligned[i]
        # Bearish alignment: price below weekly HMA
        bearish_alignment = price < hma_21_1w_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~3 days on 12h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + bullish HMA alignment
            if breakout_up and bullish_alignment:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + bearish HMA alignment
            elif breakout_down and bearish_alignment:
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