#!/usr/bin/env python3
"""
Experiment #4110: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation + chop filter
HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA(21) trend, volume confirmation, and chop regime filter capture strong trending moves while minimizing whipsaws in both bull and bear markets. Weekly timeframe adapts to regime changes, chop filter avoids range-bound false signals. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4110_1d_donchian20_1w_hma21_vol_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w HMA(21) for trend direction ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate HMA: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(df_1w['close'].values).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1w Chopiness Index(14) for regime filter ===
    if len(df_1w) >= 1:
        # True Range
        tr1 = df_1w['high'].values - df_1w['low'].values
        tr2 = np.abs(df_1w['high'].values - np.concatenate([[df_1w['close'].values[0]], df_1w['close'].values[:-1]]))
        tr3 = np.abs(df_1w['low'].values - np.concatenate([[df_1w['close'].values[0]], df_1w['close'].values[:-1]]))
        tr_w = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_w = pd.Series(tr_w).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Chopiness Index = 100 * log10(sum(ATR14) / (max(high)-min(low)) * sqrt(period))
        sum_atr_w = pd.Series(atr_w).rolling(window=14, min_periods=14).sum().values
        max_high_w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
        min_low_w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
        chop_w = 100 * np.log10(sum_atr_w / (max_high_w - min_low_w) * np.sqrt(14))
        chop_w = np.where((max_high_w - min_low_w) == 0, 50, chop_w)  # avoid div by zero
        chop_w_aligned = align_htf_to_ltf(prices, df_1w, chop_w)
    else:
        chop_w_aligned = np.full(n, 50.0)  # neutral chop
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(20) for volatility and stoploss ===
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
    
    warmup = max(lookback_dc + 1, 20 + 10, 14 + 10)  # DC lookback, vol MA buffer, chop buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(chop_w_aligned[i])):
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
        
        # Chop regime filter: only trade when trending (CHOP < 38.2) or extreme range (CHOP > 61.8)
        chop_value = chop_w_aligned[i]
        trending_regime = chop_value < 38.2
        extreme_range_regime = chop_value > 61.8
        regime_filter = trending_regime  # Focus on trending regimes for breakouts
        
        if volume_spike and regime_filter:
            # HTF 1w HMA(21) trend bias: 
            price_above_hma = price > hma_21_aligned[i]
            price_below_hma = price < hma_21_aligned[i]
            
            # Breakout logic: 
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long conditions: above 1w HMA(21) + upper Donchian breakout
            long_entry = breakout_up and price_above_hma
            
            # Short conditions: below 1w HMA(21) + lower Donchian breakout
            short_entry = breakout_down and price_below_hma
            
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