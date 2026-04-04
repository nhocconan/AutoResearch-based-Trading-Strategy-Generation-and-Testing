#!/usr/bin/env python3
"""
Experiment #2279: 6h Donchian(20) breakout + 12h Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with 12h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
and volume confirmation capture institutional order flow. Works in bull markets via breakout continuation 
and bear markets via fade at extreme pivot levels. Uses 6h primary timeframe to balance trade frequency 
and signal quality, targeting 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2279_6h_donchian20_12h_camarilla_vol_v1"
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
    
    # Calculate Camarilla pivot levels for 12h timeframe
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    camarilla_r4_12h = close_12h + (range_12h * 1.1 / 2.0)
    camarilla_r3_12h = close_12h + (range_12h * 1.1 / 4.0)
    camarilla_s3_12h = close_12h - (range_12h * 1.1 / 4.0)
    camarilla_s4_12h = close_12h - (range_12h * 1.1 / 2.0)
    
    # Trend bias: 1 if close > pivot, -1 otherwise
    trend_bias_12h = np.where(close_12h > pivot_12h, 1, -1)
    trend_bias_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_bias_12h)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
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
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_bias_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(camarilla_r3_12h[i]) or 
            np.isnan(camarilla_s3_12h[i]) or np.isnan(camarilla_r4_12h[i]) or
            np.isnan(camarilla_s4_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla S3 (mean reversion target)
                elif price <= camarilla_s3_12h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla R3 (mean reversion target)
                elif price >= camarilla_r3_12h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h trend alignment for bias filter
        trend_bias = trend_bias_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND 12h trend up
            # OR price approaches Camarilla S4 in downtrend (mean reversion long)
            if ((trend_bias > 0 and price > donchian_upper[i]) or
                (trend_bias < 0 and price <= camarilla_s4_12h[i])):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND 12h trend down
            # OR price approaches Camarilla R4 in uptrend (mean reversion short)
            elif ((trend_bias < 0 and price < donchian_lower[i]) or
                  (trend_bias > 0 and price >= camarilla_r4_12h[i])):
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