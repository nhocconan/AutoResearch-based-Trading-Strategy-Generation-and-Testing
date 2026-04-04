#!/usr/bin/env python3
"""
Experiment #5975: 6h Donchian(20) breakout + 1w Camarilla pivot levels + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with weekly Camarilla levels (R3/S3 for mean reversion,
R4/S4 for breakout continuation) capture sustained moves. Weekly Camarilla provides structural
support/resistance more reliable than daily in ranging/weak trending markets. Volume >1.5x average
confirms breakout strength. ATR trailing stop manages risk. Target 75-200 trades over 4 years.
Works in bull/bear: Camarilla levels adapt to volatility, volume confirmation avoids false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5975_6h_donchian20_1w_camarilla_vol_v1"
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
    
    # === HTF: 1w data for weekly Camarilla levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:  # Need at least 1 week for calculation
        # Calculate weekly OHLC
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Camarilla pivot levels for weekly timeframe
        # R4 = Close + ((High - Low) * 1.1/2)
        # R3 = Close + ((High - Low) * 1.1/4)
        # S3 = Close - ((High - Low) * 1.1/4)
        # S4 = Close - ((High - Low) * 1.1/2)
        camarilla_r4 = weekly_close + ((weekly_high - weekly_low) * 1.1 / 2)
        camarilla_r3 = weekly_close + ((weekly_high - weekly_low) * 1.1 / 4)
        camarilla_s3 = weekly_close - ((weekly_high - weekly_low) * 1.1 / 4)
        camarilla_s4 = weekly_close - ((weekly_high - weekly_low) * 1.1 / 2)
        
        # For breakout logic: use R4/S4 as breakout levels, R3/S3 as pullback levels
        breakout_level_up = camarilla_r4
        breakout_level_down = camarilla_s4
        
        # Align to 6h timeframe with shift(1) for completed weekly bars only
        breakout_up_aligned = align_htf_to_ltf(prices, df_1w, breakout_level_up)
        breakout_down_aligned = align_htf_to_ltf(prices, df_1w, breakout_level_down)
        
        # For mean reversion logic: use R3/S3
        mean_rev_up_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
        mean_rev_down_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    else:
        breakout_up_aligned = np.full(n, np.nan)
        breakout_down_aligned = np.full(n, np.nan)
        mean_rev_up_aligned = np.full(n, np.nan)
        mean_rev_down_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 1) + 1  # Donchian, volume avg, ATR, weekly lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                # OR mean reversion signal at R3 (weekly)
                if price <= stop_price or price <= donchian_low[i] or price <= mean_rev_up_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                # OR mean reversion signal at S3 (weekly)
                if price >= stop_price or price >= donchian_high[i] or price >= mean_rev_down_aligned[i]:
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
        
        # Weekly Camarilla levels: 
        # Breakout: price breaks above R4 or below S4 with volume
        # Mean reversion alternative: pullback to R3/S3 in opposite direction (not used for entry here)
        above_breakout = price > breakout_up_aligned[i]
        below_breakout = price < breakout_down_aligned[i]
        
        # Entry conditions: 
        # Long: Donchian breakout up with volume AND price above weekly R4 (strong breakout)
        # Short: Donchian breakout down with volume AND price below weekly S4 (strong breakout)
        long_setup = breakout_up and volume_confirmed and above_breakout
        short_setup = breakout_down and volume_confirmed and below_breakout
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals