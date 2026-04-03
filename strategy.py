#!/usr/bin/env python3
"""
Experiment #039: 6h Camarilla Pivot + Volume Spike + 12h Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h charts, 
combined with volume spikes (>2x average) and 12h EMA50 trend filter, captures institutional 
reactions at key mathematical levels. In ranging markets (price between R3-S3), we fade extremes 
with volume confirmation. In trending markets (price beyond R4/S4), we breakout in direction of 
12h trend. This adaptive approach works in both bull/bear markets by switching between mean 
reversion and trend following based on price location relative to Camarilla levels. Uses discrete 
sizing (0.25) and ATR(14) stoploss (2.5) to manage risk. Target: 100-200 total trades over 4 years 
(25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_039_6h_camarilla_vol_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for EMA50 trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === 6h Indicators: Previous period Camarilla Pivots (based on prior 6h bar) ===
    # Camarilla levels calculated from previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2.0)
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    valid_start = 20
    vol_ratio[valid_start:] = volume[valid_start:] / vol_ma[valid_start:]
    vol_ratio[:valid_start] = 1.0
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = 50  # sufficient for EMA50 and pivot calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(r4[i]) or np.isnan(s4[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price Location Relative to Camarilla Levels ---
        price_above_r4 = price > r4[i]
        price_below_s4 = price < s4[i]
        price_between_r3_s3 = (price > s3[i]) & (price < r3[i])
        price_above_r3 = price > r3[i]
        price_below_s3 = price < s3[i]
        
        # --- Trend Filter: 12h EMA50 ---
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Mean reversion fade at R3/S3 when price is between R3-S3 (ranging market)
            if price_between_r3_s3:
                if price_above_r3 and bearish_trend:  # Near R3, fading down in bearish trend
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                elif price_below_s3 and bullish_trend:  # Near S3, fading up in bullish trend
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            # Breakout continuation at R4/S4 in direction of 12h trend
            elif price_above_r4 and bullish_trend:  # Break above R4 with bullish trend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price_below_s4 and bearish_trend:  # Break below S4 with bearish trend
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