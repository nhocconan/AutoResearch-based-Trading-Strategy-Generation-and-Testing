#!/usr/bin/env python3
"""
Experiment #6371: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>1.8x average) and 1d Camarilla pivot structure capture institutional momentum. Camarilla levels provide mathematical support/resistance: R3/S3 for mean reversion fades, R4/S4 for breakout confirmation. Volume filter ensures breakout validity. Works in bull via R4 breakouts long, in bear via S4 breakdowns short with volume confirmation. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6371_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla levels from previous day
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        
        # Camarilla formula
        range_ = prev_high - prev_low
        camarilla_h4 = prev_close + range_ * 1.1 / 2  # R3
        camarilla_l4 = prev_close - range_ * 1.1 / 2  # S3
        camarilla_h5 = prev_close + range_ * 1.1      # R4
        camarilla_l5 = prev_close - range_ * 1.1      # S4
        
        # Align to 6h timeframe
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    else:
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
        camarilla_h5_aligned = np.full(n, np.nan)
        camarilla_l5_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price retraces to Camarilla H4 (take profit)
                if price <= stop_price or price <= donchian_low[i] or price <= camarilla_h4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price retraces to Camarilla L4 (take profit)
                if price >= stop_price or price >= donchian_high[i] or price >= camarilla_l4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter
        
        # Entry logic based on Camarilla levels:
        # Long: breakout above Donchian high + volume + price > Camarilla H5 (R4 breakout)
        # Short: breakdown below Donchian low + volume + price < Camarilla L5 (S4 breakdown)
        
        long_entry = breakout_up and volume_confirmed and (price > camarilla_h5_aligned[i])
        short_entry = breakout_down and volume_confirmed and (price < camarilla_l5_aligned[i])
        
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

</think>