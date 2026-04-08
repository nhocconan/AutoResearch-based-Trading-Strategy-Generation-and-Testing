#!/usr/bin/env python3
"""
Experiment #4447: 6h Donchian(20) Breakout + 1d Camarilla Pivot + Volume Confirmation + ATR Stoploss
HYPOTHESIS: 6h Donchian(20) breakouts aligned with 1d Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout) and confirmed by volume (>1.5x average) capture institutional momentum with minimal false signals. The 1d Camarilla provides key support/resistance levels from higher timeframe, reducing whipsaws in both bull and bear markets. Volume filters low-conviction moves. ATR-based trailing stop limits drawdown. Targets 75-150 total trades over 4 years (19-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4447_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Calculate pivot point
        pivot_1d = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        
        # Camarilla levels
        r3_1d = close_1d + range_1d * 1.1 / 4.0
        s3_1d = close_1d - range_1d * 1.1 / 4.0
        r4_1d = close_1d + range_1d * 1.1 / 2.0
        s4_1d = close_1d - range_1d * 1.1 / 2.0
        
        # Align to LTF (6h) with shift(1) for completed bars only
        pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
        r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
        s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
        r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
        s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    else:
        pivot_1d_aligned = np.full(n, np.nan)
        r3_1d_aligned = np.full(n, np.nan)
        s3_1d_aligned = np.full(n, np.nan)
        r4_1d_aligned = np.full(n, np.nan)
        s4_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Price relative to Camarilla levels
        price_above_r3 = price > r3_1d_aligned[i]
        price_below_s3 = price < s3_1d_aligned[i]
        price_above_r4 = price > r4_1d_aligned[i]
        price_below_s4 = price < s4_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + above R3 (bullish bias) + volume
        # OR breakout above R4 (strong breakout) + volume
        long_entry = (breakout_up and price_above_r3 and volume_confirm) or \
                     (breakout_up and price_above_r4 and volume_confirm)
        
        # Short conditions: downward breakout + below S3 (bearish bias) + volume
        # OR breakout below S4 (strong breakout) + volume
        short_entry = (breakout_down and price_below_s3 and volume_confirm) or \
                      (breakout_down and price_below_s4 and volume_confirm)
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals