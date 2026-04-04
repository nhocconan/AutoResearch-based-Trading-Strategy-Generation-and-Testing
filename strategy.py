#!/usr/bin/env python3
"""
Experiment #3207: 6h Camarilla Pivot + 1d Trend + Volume Spike
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
combined with 1d EMA trend filter and volume confirmation captures institutional 
order flow. In bull/bear markets, price respects these mathematical pivot levels 
derived from prior day's range. Volume spike (>1.8x 20-period average) confirms 
participation. Target: 75-150 total trades over 4 years (19-37/year) with 
discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3207_6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels and EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Based on prior day's OHLC: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    def calculate_camarilla(h, l, c):
        rng = h - l
        r4 = c + rng * 1.1 / 2
        r3 = c + rng * 1.1 / 4
        s3 = c - rng * 1.1 / 4
        s4 = c - rng * 1.1 / 2
        return r3, r4, s3, s4
    
    # Calculate for each 1d bar using prior day's data (shifted by 1)
    r3_1d = np.full(n, np.nan)
    r4_1d = np.full(n, np.nan)
    s3_1d = np.full(n, np.nan)
    s4_1d = np.full(n, np.nan)
    
    for i in range(1, len(high_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r3_1d[i] = r3
        r4_1d[i] = r4
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate 1d EMA(50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
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
    
    warmup = max(50, 20)  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit conditions: price reaches opposite Camarilla level or volume dries up
            if position_side > 0:  # Long
                # Exit if price reaches S3 (mean reversion target) or breaks below S4 (failed breakout)
                if price <= s3_1d_aligned[i] or price < s4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                # Exit if price reaches R3 (mean reversion target) or breaks above R4 (failed breakout)
                if price >= r3_1d_aligned[i] or price > r4_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # 1d EMA trend filter: only long above EMA, short below EMA
            price_vs_ema = price - ema_1d_aligned[i]
            
            # Long entry: price breaks above R4 with bullish 1d trend (breakout continuation)
            elif price > r4_1d_aligned[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price breaks below S4 with bearish 1d trend (breakout continuation)
            elif price < s4_1d_aligned[i] and price_vs_ema < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            # Long entry: price pulls back to S3 with bullish 1d trend (mean reversion)
            elif price <= s3_1d_aligned[i] and price_vs_ema > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: price pulls back to R3 with bearish 1d trend (mean reversion)
            elif price >= r3_1d_aligned[i] and price_vs_ema < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals