#!/usr/bin/env python3
"""
Experiment #6095: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot bias capture medium-term trends while avoiding counter-trend traps. 
Weekly pivot (calculated from prior week OHLC) provides structural support/resistance that works in both bull/bear markets. 
Volume >2.0x average confirms institutional participation. ATR(14) trailing stop (2.0x) manages risk. Discrete sizing (0.25) minimizes fee drag. 
Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6095_6h_donchian20_1w_pivot_vol_v1"
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
    
    # === HTF: 1w data for weekly pivot levels ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
        # R2 = P + (H-L), S2 = P - (H-L), R3 = H + 2*(P-L), S3 = L - 2*(H-P)
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pivot)
        
        # Align to 6h timeframe (shifted by 1 week for completed bars only)
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    else:
        pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 20) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR price breaks below S3 (failed continuation)
                if price <= stop_price or price <= s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
                # Exit: stoploss OR price breaks above R3 (failed continuation)
                if price >= stop_price or price >= r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter for stronger signals
        
        # Weekly pivot bias: price > pivot = bullish bias, price < pivot = bearish bias
        bullish_bias = price > pivot_aligned[i]
        bearish_bias = price < pivot_aligned[i]
        
        # Entry conditions:
        # Long: breakout up with volume AND bullish bias above pivot
        # Short: breakout down with volume AND bearish bias below pivot
        long_entry = breakout_up and volume_confirmed and bullish_bias
        short_entry = breakout_down and volume_confirmed and bearish_bias
        
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