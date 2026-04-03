#!/usr/bin/env python3
"""
Experiment #259: 6h Camarilla Pivot + 1d Volume Spike + ATR Regime Filter

HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
filtered by 1d volume spikes (>1.8x average) and ATR-based regime (ATR(6h)/ATR(1d) < 0.7 = low volatility = mean reversion; > 1.3 = high volatility = breakout) 
captures high-probability reversals in ranging markets and strong continuations in volatile markets. 
The 1d HTF volume spike ensures institutional participation, while the ATR regime filter 
adapts strategy to current market conditions. Targets 50-150 total trades over 4 years 
(12-37/year) with discrete position sizing to minimize fee drag. Works in both bull 
(breakouts with volume) and bear (mean reversion at extremes) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_259_6h_camarilla_1d_vol_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume and ATR regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14)
    def calculate_atr(high_arr, low_arr, close_arr, period):
        if len(high_arr) < period:
            return np.full_like(high_arr, np.nan)
        tr = np.zeros(len(high_arr))
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(high_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr
    
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume MA(20)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels from previous day ===
    # Camarilla levels: based on previous day's range
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_close_prev = np.full(n, np.nan)  # Previous day's close
    
    # Need to calculate from daily data then align to 6h
    # For each 6h bar, we need the previous 1d bar's OHLC
    # We'll compute Camarilla on 1d data then align
    if len(df_1d) >= 2:
        # Calculate Camarilla levels for each 1d bar (based on previous day)
        camarilla_h3_1d = np.full(len(df_1d), np.nan)
        camarilla_l3_1d = np.full(len(df_1d), np.nan)
        camarilla_h4_1d = np.full(len(df_1d), np.nan)
        camarilla_l4_1d = np.full(len(df_1d), np.nan)
        camarilla_close_1d = df_1d['close'].values
        
        for i in range(1, len(df_1d)):
            # Previous day's OHLC
            phigh = df_1d['high'].iloc[i-1]
            plow = df_1d['low'].iloc[i-1]
            pclose = df_1d['close'].iloc[i-1]
            range_val = phigh - plow
            
            if range_val > 0:
                camarilla_h3_1d[i] = pclose + range_val * 1.1 / 4
                camarilla_l3_1d[i] = pclose - range_val * 1.1 / 4
                camarilla_h4_1d[i] = pclose + range_val * 1.1 / 2
                camarilla_l4_1d[i] = pclose - range_val * 1.1 / 2
        
        # Align Camarilla levels to 6h timeframe
        camarilla_h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
        camarilla_l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
        camarilla_h4 = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
        camarilla_l4 = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
        camarilla_close_prev = align_htf_to_ltf(prices, df_1d, camarilla_close_1d)
    
    # === 6h Indicators: ATR(14) for stoploss and regime ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    atr_6h = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === ATR Regime: ATR(6h)/ATR(1d) ratio ===
    atr_ratio = np.zeros(n)
    valid_atr = (atr_6h > 0) & (atr_1d_aligned > 0)
    atr_ratio[valid_atr] = atr_6h[valid_atr] / atr_1d_aligned[valid_atr]
    atr_ratio[~valid_atr] = 1.0  # Neutral for invalid
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF indicators, ATR, and warmup periods
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(atr_ratio[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require 1d volume spike (> 1.8x average) ---
        volume_spike_1d = volume[i] > (vol_ma_20_1d_aligned[i] * 1.8)
        
        # --- ATR Regime Filter ---
        # Low volatility (ATR ratio < 0.7) = mean reversion regime
        # High volatility (ATR ratio > 1.3) = breakout regime
        low_vol_regime = atr_ratio[i] < 0.7
        high_vol_regime = atr_ratio[i] > 1.3
        
        # --- Camarilla Conditions ---
        # Mean reversion: fade at H3/L3 (price extreme in low vol)
        mean_rev_long = close[i] < camarilla_l3[i] and low_vol_regime
        mean_rev_short = close[i] > camarilla_h3[i] and low_vol_regime
        
        # Breakout: break H4/L4 with volume (continuation in high vol)
        breakout_long = close[i] > camarilla_h4[i] and high_vol_regime and volume_spike_1d
        breakout_short = close[i] < camarilla_l4[i] and high_vol_regime and volume_spike_1d
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_6h[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Camarilla H3 reversion (take profit for mean reversion)
                if position_side > 0 and close[i] > camarilla_h3[i] and low_vol_regime:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_6h[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Camarilla L3 reversion (take profit for mean reversion)
                if position_side < 0 and close[i] < camarilla_l3[i] and low_vol_regime:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Mean reversion at L3 OR Breakout above H4
        long_condition = mean_rev_long or breakout_long
        
        # Short: Mean reversion at H3 OR Breakdown below L4
        short_condition = mean_rev_short or breakout_short
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals