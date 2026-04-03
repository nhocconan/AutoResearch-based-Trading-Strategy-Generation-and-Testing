#!/usr/bin/env python3
"""
Experiment #115: 6h Camarilla pivot + volume confirmation + weekly trend filter

HYPOTHESIS: Camarilla pivot levels derived from 1d data provide intraday support/resistance
on the 6h chart. Fading at R3/S3 levels with volume confirmation and weekly trend filter
captures mean reversion in ranging markets while avoiding counter-trend trades. Weekly
trend filter ensures alignment with higher timeframe direction. Targets 12-37 trades/year
on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
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
    
    # Calculate Camarilla pivot levels from previous 1d bar
    if len(df_1d) >= 2:
        # Use previous day's OHLC to calculate today's Camarilla levels
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        r4 = pivot + (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1 / 2)
        
        # Align to LTF (6h) - these levels are valid for the entire day
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(21) on 1w close for trend direction
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    else:
        ema_21_1w_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # ATR(14) for volume normalization and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume ratio (current vs 20-period average)
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
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Weekly EMA21 direction ---
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at S3 (for longs fading from R3)
                if close[i] <= s3_aligned[i]:
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
                # Take profit at R3 (for shorts fading from S3)
                if close[i] >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price rejects S3 (mean reversion up) with volume and weekly uptrend
        long_condition = (
            close[i] > s3_aligned[i] and 
            low[i] <= s3_aligned[i] * 1.002 and  # Touched or slightly penetrated S3
            weekly_uptrend and 
            volume_spike
        )
        
        # Short: Price rejects R3 (mean reversion down) with volume and weekly downtrend
        short_condition = (
            close[i] < r3_aligned[i] and 
            high[i] >= r3_aligned[i] * 0.998 and  # Touched or slightly penetrated R3
            weekly_downtrend and 
            volume_spike
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