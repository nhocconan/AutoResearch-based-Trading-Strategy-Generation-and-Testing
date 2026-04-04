#!/usr/bin/env python3
"""
Experiment #5755: 6h Donchian(20) breakout + 1w Camarilla pivot regime + volume confirmation
HYPOTHESIS: Donchian breakouts aligned with weekly Camarilla pivot structure (R3/S3 for mean reversion, R4/S4 for continuation) capture sustained moves while avoiding false breakouts. Volume > 1.5x average confirms breakout strength. In bear markets (price < weekly pivot), favor shorts at R3/R4; in bull markets (price > weekly pivot), favor longs at S3/S4. This adapts to regime via pivot levels, working in both bull and bear environments. Uses 6h timeframe for lower fee drag vs lower timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5755_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1w data for Camarilla pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly Camarilla levels from previous week's OHLC
        # Using rolling window of 1 week for H, L, C
        weekly_high = df_1w['high'].rolling(window=1, min_periods=1).max().values  # current week high
        weekly_low = df_1w['low'].rolling(window=1, min_periods=1).min().values    # current week low
        weekly_close = df_1w['close'].rolling(window=1, min_periods=1).mean().values # current week close
        
        # Camarilla formula: 
        # H4 = Close + 1.5*(High-Low)
        # L4 = Close - 1.5*(High-Low)
        # H3 = Close + 1.125*(High-Low)
        # L3 = Close - 1.125*(High-Low)
        # H3 and L3 are key reversal levels
        rng = weekly_high - weekly_low
        camarilla_h3 = weekly_close + 1.125 * rng
        camarilla_l3 = weekly_close - 1.125 * rng
        camarilla_h4 = weekly_close + 1.5 * rng
        camarilla_l4 = weekly_close - 1.5 * rng
    else:
        camarilla_h3 = np.full(len(df_1w), np.nan)
        camarilla_l3 = np.full(len(df_1w), np.nan)
        camarilla_h4 = np.full(len(df_1w), np.nan)
        camarilla_l4 = np.full(len(df_1w), np.nan)
    
    # Align weekly Camarilla levels to 6h timeframe (shifted by 1 for completed weeks only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
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
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Camarilla L3 (mean reversion)
                if price <= stop_price or price <= camarilla_l3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Camarilla H3 (mean reversion)
                if price >= stop_price or price >= camarilla_h3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Camarilla regime: 
        # In bull regime (price > weekly pivot): look for longs at S3/S4, shorts at R3/R4
        # In bear regime (price < weekly pivot): look for shorts at R3/R4, longs at S3/S4
        weekly_pivot = (camarilla_h3_aligned[i] + camarilla_l3_aligned[i]) / 2
        bull_regime = price > weekly_pivot
        bear_regime = price < weekly_pivot
        
        # Long setups: breakout up with volume, aligned with regime
        long_setup_breakout = breakout_up and volume_confirmed
        long_setup_pullback = (price <= camarilla_l3_aligned[i] * 1.005) and volume_confirmed and bull_regime  # near S3 in bull
        
        # Short setups: breakout down with volume, aligned with regime
        short_setup_breakout = breakout_down and volume_confirmed
        short_setup_pullback = (price >= camarilla_h3_aligned[i] * 0.995) and volume_confirmed and bear_regime  # near R3 in bear
        
        if long_setup_breakout or long_setup_pullback:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup_breakout or short_setup_pullback:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals