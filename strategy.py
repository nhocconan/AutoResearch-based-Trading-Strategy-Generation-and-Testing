#!/usr/bin/env python3
"""
Experiment #2791: 6h Camarilla Pivot + Volume Spike + 1d Trend Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with volume confirmation and daily trend filter captures high-probability 
reversals and continuations. Works in both bull/bear markets by using 1d trend as 
regime filter. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2791_6h_camarilla_vol_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)  # 1 = uptrend, -1 = downtrend
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels from previous day ===
    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    # Camarilla levels based on previous day's range
    prev_close_1d = close_1d[:-1]  # yesterdays close
    prev_high_1d = high_1d[:-1]   # yesterdays high
    prev_low_1d = low_1d[:-1]     # yesterdays low
    
    # Calculate Camarilla levels for previous day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    camarilla_r3 = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    camarilla_s3 = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    camarilla_r4 = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 2)
    camarilla_s4 = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day for lookback)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Volume Spike Detection ===
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
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: price reaches opposite Camarilla level or volume dries up
            if position_side > 0:  # Long position
                # Exit if price reaches S3 (mean reversion target) or breaks below S4 (stop)
                if price <= camarilla_s3_aligned[i] or price < camarilla_s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if price reaches R3 (mean reversion target) or breaks above R4 (stop)
                if price >= camarilla_r3_aligned[i] or price > camarilla_r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Get daily trend bias
            trend_bias = trend_1d_aligned[i]
            
            # Long entry: price at S3/S4 with uptrend on 1d (mean reversion or breakout)
            if (abs(price - camarilla_s3_aligned[i]) < camarilla_s3_aligned[i] * 0.002 or  # Near S3
                price > camarilla_s4_aligned[i]) and trend_bias > 0:  # Break above S4 with uptrend
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price at R3/R4 with downtrend on 1d (mean reversion or breakout)
            elif (abs(price - camarilla_r3_aligned[i]) < camarilla_r3_aligned[i] * 0.002 or  # Near R3
                  price < camarilla_r4_aligned[i]) and trend_bias < 0:  # Break below R4 with downtrend
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals