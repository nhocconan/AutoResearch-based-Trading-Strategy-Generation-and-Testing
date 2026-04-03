#!/usr/bin/env python3
"""
Experiment #2211: 6h Donchian(20) breakout + 1d Camarilla pivot levels + volume confirmation
HYPOTHESIS: Donchian breakouts capture momentum while Camarilla levels (R3/S3, R4/S4) provide 
institutional support/resistance. Only trade breakouts in direction of pivot bias (R3/S3 = continuation, 
R4/S4 = reversal fade). Volume spike confirms institutional participation. 
Designed for 6h timeframe to work in both bull (trend continuation) and bear (mean reversion at extremes) markets.
Target: 75-200 total trades over 4 years (19-50/year) - optimized for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2211_6h_donchian20_1d_camarilla_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla formula based on previous day's OHLC
    camarilla_R4 = np.full(len(close_1d), np.nan)
    camarilla_R3 = np.full(len(close_1d), np.nan)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    camarilla_S4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use previous day
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Camarilla calculations
        range_val = prev_high - prev_low
        camarilla_R4[i] = prev_close + range_val * 1.1 / 2
        camarilla_R3[i] = prev_close + range_val * 1.1 / 4
        camarilla_S3[i] = prev_close - range_val * 1.1 / 4
        camarilla_S4[i] = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    
    # === 6h Indicators: Donchian(20), Volume MA(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
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
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Opposite Donchian touch or Camarilla extreme fade ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit if price touches lower Donchian (mean reversion)
                if price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla S4 (extreme - fade long)
                elif price >= camarilla_S4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit if price touches upper Donchian (mean reversion)
                if price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla R4 (extreme - fade short)
                elif price <= camarilla_R4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine pivot bias based on Camarilla levels
            # R3/S3 level: continuation bias (breakout continues)
            # R4/S4 level: reversal bias (fade the extreme)
            
            # Long entry conditions
            long_breakout = price > donchian_upper[i]
            long_continuation = long_breakout and price > camarilla_R3_aligned[i] and price < camarilla_R4_aligned[i]
            long_reversal = long_breakout and price <= camarilla_S3_aligned[i]  # Fade at S3
            
            # Short entry conditions
            short_breakout = price < donchian_lower[i]
            short_continuation = short_breakout and price < camarilla_S3_aligned[i] and price > camarilla_S4_aligned[i]
            short_reversal = short_breakout and price >= camarilla_R3_aligned[i]  # Fade at R3
            
            if long_continuation or long_reversal:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif short_continuation or short_reversal:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals