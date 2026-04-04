#!/usr/bin/env python3
"""
Experiment #2611: 6h Camarilla Pivot + Volume Spike + Regime Filter
HYPOTHESIS: Camarilla pivot levels from 1d identify institutional support/resistance. 
Fading at R3/S3 (mean reversion) and breakout continuation at R4/S4 (trend following) 
with volume confirmation and chop regime filter works in both bull (breakouts) and 
bear (mean reversion at extremes) markets. Uses 6h timeframe to balance trade frequency 
and signal quality, targeting 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2611_6h_camarilla_pivot_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # HLC = previous day's high, low, close
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_ * 1.1 / 2.0)  # R3
    s3 = pivot - (range_ * 1.1 / 2.0)  # S3
    r4 = pivot + (range_ * 1.1)        # R4
    s4 = pivot - (range_ * 1.1)        # S4
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === Regime filter: Choppiness Index from 1d ===
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low)) / log10(n)
    # Simplified: use ATR and range over 14 periods
    atr_14 = pd.Series(np.maximum(np.maximum(high_1d - low_1d, 
                                             np.abs(high_1d - np.roll(close_1d, 1))),
                                  np.abs(low_1d - np.roll(close_1d, 1)))).rolling(
        window=14, min_periods=14).mean().values
    atr_14[0:14] = np.nan
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop_raw = 100 * np.log10(atr_14 * 14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_raw[np.isnan(atr_14) | np.isnan(highest_high_14) | np.isnan(lowest_low_14)] = np.nan
    chop_6h = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_loss = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(chop_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Fixed stop loss: 2.5% adverse move
            if position_side > 0:  # Long
                if price < entry_price * 0.975:  # 2.5% stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit at R4 for longs from S3/S4, or at 5% profit
                elif price >= entry_price * 1.05 or price >= r4_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price * 1.025:  # 2.5% stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Take profit at S4 for shorts from R3/R4, or at 5% profit
                elif price <= entry_price * 0.95 or price <= s4_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Regime filter: only trade when not extreme chop (CHOP > 61.8 = ranging, < 38.2 = trending)
        # We avoid extreme chop (> 70) where whipsaws occur
        not_extreme_chop = chop_6h[i] < 70.0
        
        if volume_spike and not_extreme_chop:
            # Fade at R3/S3 (mean reversion) - sell at R3, buy at S3
            if price <= s3_6h[i] and price >= s4_6h[i]:  # Between S4 and S3 - long bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = entry_price * 0.975
                signals[i] = SIZE
            elif price >= r3_6h[i] and price <= r4_6h[i]:  # Between R3 and R4 - short bias
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = entry_price * 1.025
                signals[i] = -SIZE
            # Breakout continuation at R4/S4 (trend following)
            elif price > r4_6h[i]:  # Break above R4 - long
                in_position = True
                position_side = 1
                entry_price = close[i]
                stop_loss = entry_price * 0.975
                signals[i] = SIZE
            elif price < s4_6h[i]:  # Break below S4 - short
                in_position = True
                position_side = -1
                entry_price = close[i]
                stop_loss = entry_price * 1.025
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals