#!/usr/bin/env python3
"""
Experiment #175: 6h Camarilla Pivot + Volume Spike + Weekly Trend Filter

HYPOTHESIS: Camarilla pivot levels derived from 1d timeframe provide high-probability 
mean-reversion zones at R3/S3 and breakout continuation zones at R4/S4 on 6h timeframe. 
Filtered by weekly trend (price > weekly EMA50) to avoid counter-trend trades. Volume 
spikes (>2x average) confirm institutional participation. Targets 12-30 trades/year 
(50-120 total over 4 years) to minimize fee drag while capturing institutional 
flow in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels on 1d OHLC
    if len(df_1d) >= 1:
        # Use previous completed 1d bar for pivot calculation
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        
        # Pivot point
        pp = (prev_high + prev_low + prev_close) / 3
        # Camarilla levels
        r4 = pp + (prev_high - prev_low) * 1.1 / 2
        r3 = pp + (prev_high - prev_low) * 1.1 / 4
        s3 = pp - (prev_high - prev_low) * 1.1 / 4
        s4 = pp - (prev_high - prev_low) * 1.1 / 2
        
        # Align to 6h timeframe (auto shift(1) in align_htf_to_ltf)
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # Ensure enough data for HTF EMA50 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1w EMA50 ---
        price_above_1w_ema = close[i] > ema_50_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Level Conditions ---
        # Long: Price at S3/S4 with rejection (mean reversion) OR break above R4 (continuation)
        long_reversion = (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) or \
                         (low[i] <= s4_aligned[i] and close[i] > s4_aligned[i])
        long_breakout = close[i] > r4_aligned[i]
        
        # Short: Price at R3/R4 with rejection (mean reversion) OR break below S4 (continuation)
        short_reversion = (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) or \
                          (high[i] >= r4_aligned[i] and close[i] < r4_aligned[i])
        short_breakout = close[i] < s4_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla long condition + volume spike + price above 1w EMA50
        long_condition = (long_reversion or long_breakout) and volume_spike and price_above_1w_ema
        
        # Short: Camarilla short condition + volume spike + price below 1w EMA50
        short_condition = (short_reversion or short_breakout) and volume_spike and price_below_1w_ema
        
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