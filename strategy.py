#!/usr/bin/env python3
"""
Experiment #2327: 6h Camarilla pivot + 1d trend + volume confirmation
HYPOTHESIS: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
In uptrend (1d EMA50 > EMA200), fade at S3/S4, breakout at R3/R4. In downtrend, reverse.
Volume confirmation filters false breaks. Works in both bull/bear via trend adaptation.
Target: 75-150 total trades over 4 years (19-37/year) with discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2327_6h_camarilla1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and trend ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMAs for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    trend_bias = np.where(ema_50 > ema_200, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # Using previous day's HLC to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # first bar has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align all HTF data to 6h timeframe
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_bias)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Volume confirmation ===
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 200  # for EMA200
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic (2*ATR trailing stop using 6h ATR proxy) ---
        if in_position:
            # Estimate ATR from 6h price range (20-period average true range proxy)
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1]) if i > 0 else tr1
            tr3 = abs(low[i] - close[i-1]) if i > 0 else tr1
            atr_estimate = max(tr1, tr2, tr3)
            
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        trend = trend_aligned[i]
        volume_spike = vol_ratio[i] > 1.8  # volume confirmation
        
        if volume_spike and trend != 0:
            # In uptrend: fade at S3/S4, breakout at R3/R4
            if trend > 0:  # uptrend bias
                # Fade long at strong support (S3/S4)
                if price <= s3_aligned[i] or price <= s4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = price
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Breakout long at resistance (R3/R4) with volume
                elif price >= r3_aligned[i] or price >= r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = price
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
            # In downtrend: fade at R3/R4, breakout at S3/S4
            else:  # downtrend bias
                # Fade short at strong resistance (R3/R4)
                if price >= r3_aligned[i] or price >= r4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = price
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                # Breakdown short at support (S3/S4) with volume
                elif price <= s3_aligned[i] or price <= s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = price
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals