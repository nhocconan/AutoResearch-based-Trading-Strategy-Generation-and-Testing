#!/usr/bin/env python3
"""
Experiment #2739: 6h Camarilla Pivot Breakout with 12h Trend Filter and Volume Confirmation
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 12h/1d
combined with 6h volume spikes and 12h EMA trend filter capture institutional order flow
at key support/resistance. Works in both bull/bear by fading extremes in ranging markets
and continuing trends at extreme levels. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2739_6h_camarilla_12h_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots and EMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Calculate Camarilla pivot levels for 12h (using previous 12h bar's OHLC)
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use the previous completed 12h bar to avoid look-ahead
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    # First value will be invalid (rolled), but alignment will handle this
    
    # Calculate Camarilla levels
    camarilla_r4 = prev_close_12h + ((prev_high_12h - prev_low_12h) * 1.1 / 2)
    camarilla_r3 = prev_close_12h + ((prev_high_12h - prev_low_12h) * 1.1 / 4)
    camarilla_s3 = prev_close_12h - ((prev_high_12h - prev_low_12h) * 1.1 / 4)
    camarilla_s4 = prev_close_12h - ((prev_high_12h - prev_low_12h) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # === 6h Indicators: Volume MA for spike detection ===
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
        if (np.isnan(trend_12h_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Camarilla width as ATR proxy)
                camarilla_width = camarilla_r4_aligned[i] - camarilla_s4_aligned[i]
                atr_estimate = camarilla_width * 0.15  # approximate ATR from pivot width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below S3 (mean reversion target)
                elif price < camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                camarilla_width = camarilla_r4_aligned[i] - camarilla_s4_aligned[i]
                atr_estimate = camarilla_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above R3 (mean reversion target)
                elif price > camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h trend alignment for bias filter
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average to reduce frequency)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Mean reversion entries at extreme levels (R3/S3) - fade the move
            # Long: price drops to S3 with downtrend on 12h (expect bounce)
            if trend_bias < 0 and price <= camarilla_s3_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price rises to R3 with uptrend on 12h (expect pullback)
            elif trend_bias > 0 and price >= camarilla_r3_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            # Breakout entries at extreme levels (R4/S4) - continue the move
            # Long: price breaks above R4 with uptrend on 12h
            elif trend_bias > 0 and price >= camarilla_r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: price breaks below S4 with downtrend on 12h
            elif trend_bias < 0 and price <= camarilla_s4_aligned[i]:
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