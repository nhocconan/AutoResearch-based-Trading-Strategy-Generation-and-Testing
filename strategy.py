#!/usr/bin/env python3
"""
Experiment #3839: 6h Donchian(20) breakout + 12h Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing moves with 12h Camarilla pivot levels filtering direction (long only above R3, short only below S3). Volume (>1.5x MA20) confirms institutional participation. Works in bull markets (breakouts above R3) and bear markets (breakdowns below S3). Discrete position sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3839_6h_donchian20_12h_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1/2
    # S3 = Pivot - Range * 1.1/2
    # R4 = Pivot + Range * 1.1
    # S4 = Pivot - Range * 1.1
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r3_12h = pivot_12h + range_12h * 1.1 / 2.0
    s3_12h = pivot_12h - range_12h * 1.1 / 2.0
    r4_12h = pivot_12h + range_12h * 1.1
    s4_12h = pivot_12h - range_12h * 1.1
    
    # Align to 6h timeframe (shifted by 1 for completed bars only)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
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
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                # Calculate ATR manually for exit condition
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price < highest_since_entry - 2.5 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks below Donchian lower band (trend reversal)
                    elif price < lowest_low[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price reaches 12h S3 (mean reversion exit)
                    elif price <= s3_12h_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price > lowest_since_entry + 2.5 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks above Donchian upper band (trend reversal)
                    elif price > highest_high[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price reaches 12h R3 (mean reversion exit)
                    elif price >= r3_12h_aligned[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above 12h R3 (bullish breakout with volume confirmation)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > r3_12h_aligned[i]):    # Above 12h R3 (bullish regime)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 12h S3 (bearish breakdown with volume confirmation)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < s3_12h_aligned[i]):    # Below 12h S3 (bearish regime)
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