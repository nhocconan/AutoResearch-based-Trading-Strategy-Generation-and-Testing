#!/usr/bin/env python3
"""
Experiment #403: 4h Donchian(20) + 12h Volume Spike + 1d HMA Trend + Choppiness Filter

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 12h volume spike (>2.0x average) 
and aligned with 1d HMA(21) trend, filtered by choppiness regime (CHOP(14) < 38.2 = trending), 
produces high-probability trades with minimal false signals. Uses discrete position sizing (0.30) 
and ATR(14) stoploss (2.5x) to manage risk. Targets 20-50 trades/year on 4h timeframe 
(80-200 total over 4 years) to minimize fee drag while capturing strong directional moves. 
Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets via 
symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_vol_1d_hma_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = len(close_1d) // 2
        sqrt_n = int(np.sqrt(len(close_1d)))
        wma_half = pd.Series(close_1d).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(close_1d).rolling(window=len(close_1d), min_periods=len(close_1d)).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for choppiness regime (Call ONCE before loop) ===
    # Chopiness Index: log(sum(TR14)/ (HHV(high,14) - LLV(low,14))) * 100 / log(14)
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = np.zeros(len(high_1d))
        tr1[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(high_1d)):
            tr1[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Sum of TR14
        tr_sum_14 = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
        
        # HHV and LLV of 14 periods
        hh_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        range_14 = hh_high_14 - ll_low_14
        
        # Chopiness Index
        chop_1d = np.zeros(len(close_1d))
        mask = (tr_sum_14 > 0) & (range_14 > 0)
        chop_1d[mask] = 100 * np.log10(tr_sum_14[mask] / range_14[mask]) / np.log10(14)
        chop_1d[~mask] = 50.0  # Neutral when undefined
        
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, 50.0)
    
    # === 4h Indicators ===
    # Donchian Channel (20)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(hma_21_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        # 1. Trending market: Chopiness Index < 38.2
        is_trending = chop_1d_aligned[i] < 38.2
        
        # 2. Trend alignment: price vs 1d HMA21
        price_above_hma = close[i] > hma_21_1d_aligned[i]
        price_below_hma = close[i] < hma_21_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian Low (trailing stop)
                if close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian High (trailing stop)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian High with volume spike and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_hma and 
            is_trending
        )
        
        # Short: Price breaks below Donchian Low with volume spike and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_hma and 
            is_trending
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals