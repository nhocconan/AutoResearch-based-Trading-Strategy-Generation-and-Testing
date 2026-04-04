#!/usr/bin/env python3
"""
Experiment #2511: 6h Camarilla Pivot + Volume Spike + Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with volume confirmation and 1d trend filter captures institutional 
participation at key levels. Works in both bull and bear markets by fading extremes 
in ranging conditions and continuing breaks in trending markets. Targets 75-150 
total trades over 4 years with discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2511_6h_camarilla1d_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    #          S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    camarilla_r4 = typical_1d + (range_1d * 1.1 / 2.0)
    camarilla_r3 = typical_1d + (range_1d * 1.1 / 4.0)
    camarilla_s3 = typical_1d - (range_1d * 1.1 / 4.0)
    camarilla_s4 = typical_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                # Approximate ATR from 6h range
                atr_estimate = (high[i] - low[i]) * 0.15
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla S3 (mean reversion target) or breaks S4 (stop)
                elif price <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price < s4_aligned[i]:  # Stop loss if breaks S4
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                atr_estimate = (high[i] - low[i]) * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla R3 (mean reversion target) or breaks R4 (stop)
                elif price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                elif price > r4_aligned[i]:  # Stop loss if breaks R4
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and trend_bias != 0:
            # Mean reversion entries at R3/S3 in ranging/weak trend conditions
            # Long: price drops to S3 with volume spike in uptrend
            if trend_bias > 0 and price <= s3_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price rises to R3 with volume spike in downtrend
            elif trend_bias < 0 and price >= r3_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Breakout entries at R4/S4 with strong trend alignment
            # Long: price breaks above R4 with volume spike in uptrend
            elif trend_bias > 0 and price >= r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price breaks below S4 with volume spike in downtrend
            elif trend_bias < 0 and price <= s4_aligned[i]:
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