#!/usr/bin/env python3
"""
Experiment #4111: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla pivot structure (R3/S3 for mean reversion, R4/S4 for breakout) and volume confirmation capture institutional order flow. Works in bull/bear by using pivot levels as dynamic support/resistance. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4111_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate Camarilla pivot points from previous day
        # PP = (H + L + C) / 3
        # R4 = PP + (H - L) * 1.1/2
        # R3 = PP + (H - L) * 1.1/4
        # S3 = PP - (H - L) * 1.1/4
        # S4 = PP - (H - L) * 1.1/2
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        
        pp = (h_1d + l_1d + c_1d) / 3.0
        r4 = pp + (h_1d - l_1d) * 1.1 / 2.0
        r3 = pp + (h_1d - l_1d) * 1.1 / 4.0
        s3 = pp - (h_1d - l_1d) * 1.1 / 4.0
        s4 = pp - (h_1d - l_1d) * 1.1 / 2.0
        
        # Align to 6h timeframe (shifted by 1 for completed bars only)
        pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    else:
        pp_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i])):
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
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Camarilla pivot logic:
            # R3/S3: mean reversion zone (fade extremes)
            # R4/S4: breakout zone (continuation)
            near_r3 = abs(price - r3_aligned[i]) / r3_aligned[i] < 0.005  # within 0.5%
            near_s3 = abs(price - s3_aligned[i]) / s3_aligned[i] < 0.005
            breakout_r4 = price > r4_aligned[i]
            breakdown_s4 = price < s4_aligned[i]
            
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long conditions:
            # 1. Breakout above R4 with Donchian breakout (continuation)
            # 2. Mean reversion from S3 with Donchian breakout (bounce)
            long_continuation = breakout_r4 and breakout_up
            long_mean_reversion = near_s3 and breakout_up
            long_entry = long_continuation or long_mean_reversion
            
            # Short conditions:
            # 1. Breakdown below S4 with Donchian breakdown (continuation)
            # 2. Mean reversion from R3 with Donchian breakdown (rejection)
            short_continuation = breakdown_s4 and breakout_down
            short_mean_reversion = near_r3 and breakout_down
            short_entry = short_continuation or short_mean_reversion
            
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