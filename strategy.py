#!/usr/bin/env python3
"""
Experiment #1699: 6h Camarilla Pivot Fade/Breakout + 12h Trend + Volume Spike
HYPOTHESIS: Camarilla pivot levels from 12h timeframe provide statistically significant support/resistance. 
Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels (trend following) 
with 12h EMA20 trend filter and volume confirmation (>1.8x average) captures both range-bound 
and trending markets. 6h timeframe balances trade frequency and signal quality. Target: 75-150 
total trades over 4 years (19-38/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1699_6h_camarilla12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots and trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    #                S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    r4_12h = close_12h + (range_12h * 1.1 / 2.0)
    r3_12h = close_12h + (range_12h * 1.1 / 4.0)
    s3_12h = close_12h - (range_12h * 1.1 / 4.0)
    s4_12h = close_12h - (range_12h * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 12h EMA20 for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter: price above/below 12h EMA20
        uptrend = price > ema_12h_aligned[i]
        downtrend = price < ema_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Fade at R3/S3 (mean reversion) - counter-trend
            if price <= r3_12h_aligned[i] and not uptrend:  # At R3 in downtrend -> short
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif price >= s3_12h_aligned[i] and not downtrend:  # At S3 in uptrend -> long
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Breakout continuation at R4/S4 (trend following)
            elif price > r4_12h_aligned[i] and uptrend:  # Break above R4 in uptrend -> long
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < s4_12h_aligned[i] and downtrend:  # Break below S4 in downtrend -> short
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
</file>