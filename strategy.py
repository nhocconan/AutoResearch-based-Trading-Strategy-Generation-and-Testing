#!/usr/bin/env python3
"""
Experiment #4287: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum when aligned with 1d Camarilla pivot structure (long at S3/S4 support, short at R3/R4 resistance) and confirmed by volume (>1.8x average). Uses 6h primary timeframe to balance noise and trade frequency, targeting 75-150 total trades over 4 years (19-38/year). ATR trailing stop (2.0x) manages risk. Works in bull via breakout continuation at R4, in bear via breakdown continuation at S4. Novelty: Camarilla pivot levels from 1d provide institutional support/resistance that works across regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4287_6h_donchian20_1d_camarilla_vol_v1"
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
        # Calculate Camarilla pivot levels from previous day's OHLC
        # R4 = Close + ((High - Low) * 1.5)
        # R3 = Close + ((High - Low) * 1.25)
        # R2 = Close + ((High - Low) * 1.166)
        # R1 = Close + ((High - Low) * 1.083)
        # PP = (High + Low + Close) / 3
        # S1 = Close - ((High - Low) * 1.083)
        # S2 = Close - ((High - Low) * 1.166)
        # S3 = Close - ((High - Low) * 1.25)
        # S4 = Close - ((High - Low) * 1.5)
        
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        # Calculate levels for each day
        camarilla_r4 = np.zeros_like(c_1d)
        camarilla_r3 = np.zeros_like(c_1d)
        camarilla_s3 = np.zeros_like(c_1d)
        camarilla_s4 = np.zeros_like(c_1d)
        
        for i in range(len(c_1d)):
            rng = h_1d[i] - l_1d[i]
            camarilla_r4[i] = c_1d[i] + (rng * 1.5)
            camarilla_r3[i] = c_1d[i] + (rng * 1.25)
            camarilla_s3[i] = c_1d[i] - (rng * 1.25)
            camarilla_s4[i] = c_1d[i] - (rng * 1.5)
        
        # Align to 6h timeframe (use previous day's levels)
        camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        camarilla_r4_aligned = np.full(n, np.nan)
        camarilla_r3_aligned = np.full(n, np.nan)
        camarilla_s3_aligned = np.full(n, np.nan)
        camarilla_s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
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
            np.isnan(atr[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
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
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.8x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.8
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Camarilla pivot conditions
            near_r3_r4 = (price >= camarilla_r3_aligned[i] * 0.998) or (price >= camarilla_r4_aligned[i] * 0.995)
            near_s3_s4 = (price <= camarilla_s3_aligned[i] * 1.002) or (price <= camarilla_s4_aligned[i] * 1.005)
            
            # Long conditions: Donchian breakout up + near R3/R4 resistance (breakout continuation)
            long_entry = breakout_up and near_r3_r4
            
            # Short conditions: Donchian breakout down + near S3/S4 support (breakdown continuation)
            short_entry = breakout_dn and near_s3_s4
            
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
        else:
            signals[i] = 0.0
    
    return signals