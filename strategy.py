#!/usr/bin/env python3
"""
Experiment #6395: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>2.0x avg) and weekly Camarilla pivot filter capture institutional order flow. Weekly R3/S3 levels act as mean-reversion zones (fade), while R4/S4 levels indicate breakout strength (continuation). This structure works in bull markets via R4 breakout continuation and in bear markets via S4 breakdown continuation, with volume filtering false signals. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6395_6h_donchian20_1w_camarilla_vol_v1"
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
    if len(df_1w) >= 2:
        # Calculate weekly Camarilla levels from prior week's OHLC
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Camarilla formula: R4 = Close + (High-Low)*1.1/2, R3 = Close + (High-Low)*1.1/4, etc.
        camarilla_r4 = weekly_close + (weekly_high - weekly_low) * 1.1 / 2
        camarilla_r3 = weekly_close + (weekly_high - weekly_low) * 1.1 / 4
        camarilla_s3 = weekly_close - (weekly_high - weekly_low) * 1.1 / 4
        camarilla_s4 = weekly_close - (weekly_high - weekly_low) * 1.1 / 2
        
        # Align to 6h timeframe (shift by 1 for completed weekly bars only)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
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
                # 3. Price retraces to weekly S3 (mean reversion in range)
                if price <= stop_price or price <= donchian_low[i] or price <= camarilla_s3_aligned[i]:
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
                # 3. Price retraces to weekly R3 (mean reversion in range)
                if price >= stop_price or price >= donchian_high[i] or price >= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter
        
        # Entry logic based on weekly Camarilla levels:
        # Long: breakout up + volume + price > weekly R4 (strong continuation)
        # Short: breakout down + volume + price < weekly S4 (strong breakdown)
        # Note: Avoid fading at R3/S3 unless in extreme overbought/oversold (not implemented here for simplicity)
        
        long_entry = breakout_up and volume_confirmed and (price > camarilla_r4_aligned[i])
        short_entry = breakout_down and volume_confirmed and (price < camarilla_s4_aligned[i])
        
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