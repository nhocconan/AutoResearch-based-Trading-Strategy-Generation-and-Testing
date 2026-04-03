#!/usr/bin/env python3
"""
Experiment #305: 12h Camarilla Pivot Breakout + 1d Volume Spike + Choppiness Regime Filter

HYPOTHESIS: 12-hour Camarilla pivot levels (derived from 1d OHLC) act as institutional support/resistance. 
Breakouts above R4 or below S4 with volume confirmation (>2x average) and choppiness regime filter 
(CHOP > 61.8 = ranging market) capture high-probability mean-reversion trades in ranging markets, 
while avoiding false breakouts in strong trends. The 12h timeframe targets 12-37 trades/year (50-150 total) 
to minimize fee drag. Works in bull markets (fading overextended moves at pivots) and bear markets 
(buying panic selling at S3/S4, selling rallies at R3/R4). Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_305_12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    camarilla_p = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        camarilla_p[i] = (h + l + c) / 3
        rang = h - l
        camarilla_h4[i] = c + rang * 1.1 / 2
        camarilla_l4[i] = c - rang * 1.1 / 2
        camarilla_h3[i] = c + rang * 1.1 / 4
        camarilla_l3[i] = c - rang * 1.1 / 4
        camarilla_h2[i] = c + rang * 1.1 / 6
        camarilla_l2[i] = c - rang * 1.1 / 6
        camarilla_h1[i] = c + rang * 1.1 / 12
        camarilla_l1[i] = c - rang * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 for completed 1d bar only)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_12h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_12h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_12h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_12h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    p_12h = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 12h Indicators: Choppiness Index (CHOP) for regime filter ===
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    def choppiness_index(high, low, close, period=14):
        if len(high) < period:
            return np.full_like(high, np.nan)
        tr_sum = np.zeros(len(high))
        for i in range(len(high)):
            if i == 0:
                tr_sum[i] = true_range(high[i], low[i], close[i])
            else:
                tr_sum[i] = tr_sum[i-1] + true_range(high[i], low[i], close[i-1])
        
        max_high = np.zeros(len(high))
        min_low = np.zeros(len(high))
        max_high[0] = high[0]
        min_low[0] = low[0]
        for i in range(1, len(high)):
            max_high[i] = max(max_high[i-1], high[i])
            min_low[i] = min(min_low[i-1], low[i])
        
        chop = np.zeros(len(high))
        for i in range(period-1, len(high)):
            tr_period = tr_sum[i] - (tr_sum[i-period] if i >= period else 0)
            max_high_period = max_high[i] - (max_high[i-period] if i >= period else 0)
            min_low_period = min_low[i] - (min_low[i-period] if i >= period else 0)
            range_period = max_high_period - min_low_period
            if range_period > 0 and tr_period > 0:
                chop[i] = 100 * np.log10(tr_period / range_period) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Choppiness Regime Filter: Only trade in ranging markets (CHOP > 61.8) ---
        ranging_market = chop[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 2x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = close[i] > h4_12h[i]   # Break above R4
        breakout_down = close[i] < l4_12h[i]  # Break below S4
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on pivot mean reversion (take profit at P)
                if close[i] < p_12h[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on pivot mean reversion (take profit at P)
                if close[i] > p_12h[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Breakout above R4 + volume spike + ranging market
        long_condition = breakout_up and volume_spike and ranging_market
        
        # Short: Breakout below S4 + volume spike + ranging market
        short_condition = breakout_down and volume_spike and ranging_market
        
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