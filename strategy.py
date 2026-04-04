#!/usr/bin/env python3
"""
Experiment #6115: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly Camarilla pivot levels (R4/S4 for continuation, R3/S3 for mean reversion) capture medium-term trends with confluence. Weekly pivot provides structural support/resistance effective in both bull and bear markets. Volume >1.8x average confirms strong participation. ATR(14) trailing stop (2.5x) manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 75-150 trades over 4 years.
Timeframe: 6h. HTF: 1w for Camarilla pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6115_6h_donchian20_1w_camarilla_vol_v1"
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
    if len(df_1w) >= 5:  # Need at least one weekly bar
        # Calculate weekly Camarilla levels from previous week's OHLC
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Camarilla levels: based on previous week's range
        weekly_range = weekly_high - weekly_low
        camarilla_h5 = weekly_close + 1.1 * weekly_range * 1.1 / 2  # R4 equivalent
        camarilla_h4 = weekly_close + 1.1 * weekly_range * 1.1 / 4  # R3 equivalent
        camarilla_l4 = weekly_close - 1.1 * weekly_range * 1.1 / 4  # S3 equivalent
        camarilla_l5 = weekly_close - 1.1 * weekly_range * 1.1 / 2  # S4 equivalent
        
        # Align to 6h timeframe (shift by 1 to avoid look-ahead)
        camarilla_h5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h5)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
        camarilla_l5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l5)
    else:
        camarilla_h5_aligned = np.full(n, np.nan)
        camarilla_h4_aligned = np.full(n, np.nan)
        camarilla_l4_aligned = np.full(n, np.nan)
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
    
    warmup = max(20, 20, 5, 14) + 1  # Donchian, volume avg, weekly data, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below weekly S4 (failed continuation)
                if price <= stop_price or price <= camarilla_l5_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above weekly R4 (failed continuation)
                if price >= stop_price or price >= camarilla_h5_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter for stronger signals
        
        # Multi-timeframe confluence: 
        # Long: breakout above weekly R4 (continuation) OR pullback to weekly S3 (mean reversion in uptrend)
        # Short: breakout below weekly S4 (continuation) OR pullback to weekly R3 (mean reversion in downtrend)
        long_continuation = breakout_up and price > camarilla_h5_aligned[i]
        long_mean_reversion = breakout_down and price > camarilla_l4_aligned[i] and price < camarilla_h4_aligned[i]
        short_continuation = breakout_down and price < camarilla_l5_aligned[i]
        short_mean_reversion = breakout_up and price < camarilla_h4_aligned[i] and price > camarilla_l4_aligned[i]
        
        long_entry = (long_continuation or long_mean_reversion) and volume_confirmed
        short_entry = (short_continuation or short_mean_reversion) and volume_confirmed
        
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