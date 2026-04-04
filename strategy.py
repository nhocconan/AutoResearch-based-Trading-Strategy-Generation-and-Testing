#!/usr/bin/env python3
"""
Experiment #3159: 6h Donchian(20) breakout + 12h Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture medium-term momentum with controlled frequency. 
12h Camarilla pivot levels (R3/S3 for mean reversion fade, R4/S4 for breakout continuation) 
provide institutional reference points. Volume spike (>2.0x 20-period average) confirms 
strength. ATR trailing stop (2.5x) manages risk. Position size 0.25. Target: 75-200 total 
trades over 4 years (19-50/year). Designed for both bull (breakout continuation at R4/S4) 
and bear (mean reversion fade at R3/S3) markets by using Camarilla levels as dynamic 
support/resistance with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3159_6h_donchian20_12h_camarilla_vol_v1"
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
    
    # Calculate Camarilla levels for 12h: based on previous day's range
    # Camarilla: R4 = close + (high-low)*1.1/2, R3 = close + (high-low)*1.1/4
    #            S3 = close - (high-low)*1.1/4, S4 = close - (high-low)*1.1/2
    # We use the completed 12h bar's range to calculate levels for current 6h bar
    prev_high_12h = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low_12h = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    
    range_12h = prev_high_12h - prev_low_12h
    camarilla_r4 = prev_close_12h + range_12h * 1.1 / 2
    camarilla_r3 = prev_close_12h + range_12h * 1.1 / 4
    camarilla_s3 = prev_close_12h - range_12h * 1.1 / 4
    camarilla_s4 = prev_close_12h - range_12h * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Camarilla-based logic:
            # Long: breakout above Donchian high AND price > R3 (fade resistance) OR break above R4 (continuation)
            # Short: breakdown below Donchian low AND price < S3 (fade support) OR break below S4 (continuation)
            
            # Long conditions
            long_breakout = price > highest_high[i]
            long_fade = price > camarilla_r3_aligned[i]  # Above R3 = bullish bias for fade
            long_continuation = price > camarilla_r4_aligned[i]  # Above R4 = strong breakout
            
            # Short conditions
            short_breakout = price < lowest_low[i]
            short_fade = price < camarilla_s3_aligned[i]  # Below S3 = bearish bias for fade
            short_continuation = price < camarilla_s4_aligned[i]  # Below S4 = strong breakdown
            
            if long_breakout and (long_fade or long_continuation):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout and (short_fade or short_continuation):
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