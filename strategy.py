#!/usr/bin/env python3
"""
Experiment #1579: 6h Donchian(20) Breakout + 12h Camarilla Pivot + Volume Spike
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Camarilla pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) and volume confirmation (>2.0x average) capture institutional breakouts while avoiding false signals. The 12h timeframe provides structural pivot levels, while 6h Donchian offers precise entry timing. Target: 75-150 total trades over 4 years (19-37/year) by using tight entry conditions (volume > 2.0x, Camarilla level alignment) and ATR-based stoploss.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1579_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla pivot calculation for 12h timeframe
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4_12h = close_12h + (range_12h * 1.1 / 2.0)
    r3_12h = close_12h + (range_12h * 1.1 / 4.0)
    s3_12h = close_12h - (range_12h * 1.1 / 4.0)
    s4_12h = close_12h - (range_12h * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(pivot_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Breakout logic: price breaks above R4 OR below S4 with continuation
            if price > r4_12h_aligned[i] and price > donch_high[i]:
                # Strong upward breakout above R4
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < s4_12h_aligned[i] and price < donch_low[i]:
                # Strong downward breakdown below S4
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Mean reversion logic: price touches R3/S3 and reverses
            elif abs(price - r3_12h_aligned[i]) < (r4_12h_aligned[i] - r3_12h_aligned[i]) * 0.02:
                # Near R3, potential reversal down
                if price < close_12h[-1] if len(close_12h) > 0 else price < close[i]:  # Price below prior close
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif abs(price - s3_12h_aligned[i]) < (s3_12h_aligned[i] - s4_12h_aligned[i]) * 0.02:
                # Near S3, potential reversal up
                if price > close_12h[-1] if len(close_12h) > 0 else price > close[i]:  # Price above prior close
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals