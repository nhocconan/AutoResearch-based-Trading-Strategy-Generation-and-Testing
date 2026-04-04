#!/usr/bin/env python3
"""
Experiment #4299: 6h Donchian(20) breakout + 12h Camarilla pivot + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h timeframe capture swing momentum when aligned with 12h Camarilla pivot levels (breakout at R4/S4 for continuation, fade at R3/S3 for mean reversion) and confirmed by volume (>2.0x average). Uses 12h HTF for pivot calculation to reduce noise while targeting 50-150 total trades over 4 years (12-37/year). ATR-based trailing stop (2.5x) for risk management. Position size 0.25 balances profit potential and drawdown control. Works in bull via breakout continuation at R4/S4, in bear via shorting breakdowns at R4/S4 and mean reversion fades at R3/S3. Novelty: Combines Donchian structure with Camarilla pivot levels on 6h/12h timeframe, a genuinely novel approach not recently tried in this experiment series.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4299_6h_donchian20_12h_camarilla_vol_v1"
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
    
    # === Precompute HTF: 12h Camarilla pivot levels ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla pivot levels from previous 12h bar
        # PP = (H + L + C) / 3
        # R4 = PP + (H - L) * 1.1/2
        # R3 = PP + (H - L) * 1.1/4
        # S3 = PP - (H - L) * 1.1/4
        # S4 = PP - (H - L) * 1.1/2
        pp_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
        range_12h = df_12h['high'].values - df_12h['low'].values
        r4_12h = pp_12h + range_12h * 1.1 / 2
        r3_12h = pp_12h + range_12h * 1.1 / 4
        s3_12h = pp_12h - range_12h * 1.1 / 4
        s4_12h = pp_12h - range_12h * 1.1 / 2
        
        # Align to 6h timeframe (shifted by 1 for completed bars only)
        r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
        r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
        s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
        s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    else:
        r4_12h_aligned = np.full(n, np.nan)
        r3_12h_aligned = np.full(n, np.nan)
        s3_12h_aligned = np.full(n, np.nan)
        s4_12h_aligned = np.full(n, np.nan)
    
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
            np.isnan(atr[i]) or np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i])):
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Camarilla pivot conditions from 12h
            at_r4 = abs(price - r4_12h_aligned[i]) < (r4_12h_aligned[i] - s4_12h_aligned[i]) * 0.02  # Near R4
            at_s4 = abs(price - s4_12h_aligned[i]) < (r4_12h_aligned[i] - s4_12h_aligned[i]) * 0.02  # Near S4
            at_r3 = abs(price - r3_12h_aligned[i]) < (r3_12h_aligned[i] - s3_12h_aligned[i]) * 0.02  # Near R3
            at_s3 = abs(price - s3_12h_aligned[i]) < (r3_12h_aligned[i] - s3_12h_aligned[i]) * 0.02  # Near S3
            
            # Long conditions:
            # 1. Donchian breakout up + at/above R4 (breakout continuation)
            # 2. Donchian breakout down + at/below S3 (mean reversion fade)
            long_entry = (breakout_up and (at_r4 or price > r4_12h_aligned[i])) or \
                         (breakout_dn and (at_s3 or price < s3_12h_aligned[i]))
            
            # Short conditions:
            # 1. Donchian breakout down + at/below S4 (breakout continuation)
            # 2. Donchian breakout up + at/above R3 (mean reversion fade)
            short_entry = (breakout_dn and (at_s4 or price < s4_12h_aligned[i])) or \
                          (breakout_up and (at_r3 or price > r3_12h_aligned[i]))
            
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