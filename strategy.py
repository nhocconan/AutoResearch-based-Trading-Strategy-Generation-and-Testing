#!/usr/bin/env python3
"""
Experiment #092: 12h Camarilla pivot + volume spike + choppiness regime

HYPOTHESIS: Camarilla pivot levels from 1d timeframe act as intraday support/resistance. 
Price reacting to these levels with volume confirmation during low-chop regimes (trending markets) 
captures high-probability breakout/mean-reversion opportunities. 12h timeframe reduces noise and 
overtrading. Choppiness filter ensures we only trade when market is trending (CHOP < 38.2) or 
deeply choppy (CHOP > 61.8) for mean reversion, avoiding whipsaw in transitional regimes.
Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) >= 2:
        # Camarilla pivot levels from previous 1d bar
        # Using previous day's OHLC to avoid look-ahead
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        pivot = (high_1d + low_1d + close_1d) / 3
        range_1d = high_1d - low_1d
        
        # Camarilla levels
        l3 = pivot + (range_1d * 1.1 / 4)
        l4 = pivot + (range_1d * 1.1 / 2)
        h3 = pivot - (range_1d * 1.1 / 4)
        h4 = pivot - (range_1d * 1.1 / 2)
        
        # Align to 12h timeframe (using previous day's levels)
        l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
        l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
        h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
        h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    else:
        l3_aligned = l4_aligned = h3_aligned = h4_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 12h Indicators ===
    # Choppiness Index (14) - measures trend vs range
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr = np.zeros(len(close))
        for i in range(len(close)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = tr[i] if i < period else (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        sum_atr = np.zeros(len(close))
        for i in range(len(close)):
            if i < period:
                sum_atr[i] = np.sum(atr[:i+1])
            else:
                sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros(len(close))
        lowest_low = np.zeros(len(close))
        for i in range(len(close)):
            start_idx = max(0, i - period + 1)
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
        
        # Chop formula: 100 * log10(sum(ATR) / (HH - LL)) / log10(period)
        hh_ll = highest_high - lowest_low
        chop = np.zeros(len(close))
        for i in range(len(close)):
            if hh_ll[i] > 0:
                chop[i] = 100 * np.log10(sum_atr[i] / hh_ll[i]) / np.log10(period)
            else:
                chop[i] = 50  # Neutral when no range
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Choppiness Index ---
        # CHOP < 38.2 = strong trend (trend following)
        # CHOP > 61.8 = strong range (mean reversion)
        # 38.2 <= CHOP <= 61.8 = transitional (avoid)
        chop_value = chop[i]
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) on 1w ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] >= h4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] <= l4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Regime-dependent entry logic
        if is_trending:
            # Trending market: breakout Camarilla levels
            long_condition = (
                close[i] > h4_aligned[i] and 
                volume_spike
            )
            
            short_condition = (
                close[i] < l4_aligned[i] and 
                volume_spike
            )
        elif is_ranging:
            # Ranging market: mean reversion at extreme levels
            long_condition = (
                close[i] <= l3_aligned[i] and 
                close[i] > l4_aligned[i] and  # Avoid catching falling knife
                volume_spike
            )
            
            short_condition = (
                close[i] >= h3_aligned[i] and 
                close[i] < h4_aligned[i] and  # Avoid catching rising spike
                volume_spike
            )
        else:
            # Transitional regime: no trading
            long_condition = False
            short_condition = False
        
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