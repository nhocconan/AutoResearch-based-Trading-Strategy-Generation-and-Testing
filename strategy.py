#!/usr/bin/env python3
"""
Experiment #5831: 6h Camarilla pivot levels from 1d with volume confirmation
HYPOTHESIS: 6h price reactions at 1d Camarilla R3/S3 (fade) and R4/S4 (breakout) capture institutional order flow. 
In ranging markets, price respects R3/S3 for mean reversion. In trending markets, breaks of R4/S4 with volume 
confirmation indicate strong momentum continuation. Works in bull markets (R4 breakouts with volume) and bear 
markets (S4 breakdowns with volume). Uses discrete position sizing (0.25) to minimize fee churn. Targets 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5831_6h_camarilla1d_vol_v1"
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
    
    # === HTF: 1d data for Camarilla pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate Camarilla levels from prior day's OHLC
        # H = high, L = low, C = close of previous day
        h_1d = df_1d['high'].shift(1).values  # previous day high
        l_1d = df_1d['low'].shift(1).values   # previous day low
        c_1d = df_1d['close'].shift(1).values # previous day close
        
        # Camarilla calculations
        # R4 = C + ((H-L) * 1.1/2)
        # R3 = C + ((H-L) * 1.1/4)
        # S3 = C - ((H-L) * 1.1/4)
        # S4 = C - ((H-L) * 1.1/2)
        camarilla_r4 = c_1d + ((h_1d - l_1d) * 1.1 / 2)
        camarilla_r3 = c_1d + ((h_1d - l_1d) * 1.1 / 4)
        camarilla_s3 = c_1d - ((h_1d - l_1d) * 1.1 / 4)
        camarilla_s4 = c_1d - ((h_1d - l_1d) * 1.1 / 2)
        
        # Align to 6h timeframe (shift(1) already applied above for prior day)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 14)  # volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below S3 (failed hold) OR reaches R4 (take profit)
                if price <= stop_price or price <= camarilla_s3_aligned[i] or price >= camarilla_r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above R3 (failed hold) OR reaches S4 (take profit)
                if price >= stop_price or price >= camarilla_r3_aligned[i] or price <= camarilla_s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Fade at R3/S3: price rejects these levels
        fade_r3 = (abs(price - camarilla_r3_aligned[i]) < (0.001 * camarilla_r3_aligned[i])) and (high[i] < camarilla_r3_aligned[i])
        fade_s3 = (abs(price - camarilla_s3_aligned[i]) < (0.001 * camarilla_s3_aligned[i])) and (low[i] > camarilla_s3_aligned[i])
        
        # Breakout continuation at R4/S4: price breaks these levels with volume
        breakout_r4 = price > camarilla_r4_aligned[i]
        breakout_s4 = price < camarilla_s4_aligned[i]
        
        # Entry conditions
        long_setup = (fade_r3 or breakout_r4) and volume_confirmed
        short_setup = (fade_s3 or breakout_s4) and volume_confirmed
        
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