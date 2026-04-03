#!/usr/bin/env python3
"""
Experiment #1939: 6h Donchian Breakout + 12h Volume Regime + 1d ATR Filter
HYPOTHESIS: Donchian(20) breakouts on 6h timeframe capture momentum bursts. 
Filtered by 12h volume regime (high volume = institutional participation) and 
1d ATR normalization to avoid breakouts during low volatility periods. 
Works in bull/bear markets by only taking breakouts with volume confirmation 
and avoiding choppy conditions. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1939_6h_donchian20_12h_volreg_1d_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume regime (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    
    # Volume regime: ratio of current volume to 24-period MA (48h lookback)
    vol_ma_12h = pd.Series(vol_12h).rolling(window=24, min_periods=24).mean().values
    vol_ratio_12h = np.ones(len(vol_12h))
    vol_ratio_12h[24:] = vol_12h[24:] / vol_ma_12h[24:]
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === HTF: 1d data for ATR filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) on 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 6h Indicators: Donchian(20) channels ===
    # Donchian upper = highest high of past 20 bars
    # Donchian lower = lowest low of past 20 bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based trailing stop ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate dynamic stop loss based on ATR
            atr_val = atr_1d_aligned[i]
            if atr_val <= 0:  # Avoid division by zero
                atr_val = 0.001 * price  # Fallback to 0.1% of price
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Trailing stop: highest high since entry minus 2*ATR
                # We approximate by using current Donchian lower as dynamic support
                if price <= donchian_lower[i]:
                    exit_signal = True
                # Also exit if price drops more than 2*ATR from entry
                elif price < entry_price - 2.0 * atr_val:
                    exit_signal = True
            else:  # Short position
                # Trailing stop: lowest low since entry plus 2*ATR
                # We approximate by using current Donchian upper as dynamic resistance
                if price >= donchian_upper[i]:
                    exit_signal = True
                # Also exit if price rises more than 2*ATR from entry
                elif price > entry_price + 2.0 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume regime filter: require elevated volume (> 1.3x average)
        volume_regime = vol_ratio_12h_aligned[i] > 1.3
        
        # ATR filter: require sufficient volatility (avoid breakouts in choppy low-vol periods)
        # Normalize ATR by price to get percentage volatility
        vol_percent = atr_1d_aligned[i] / price if price > 0 else 0
        sufficient_vol = vol_percent > 0.01  # At least 1% daily volatility
        
        if volume_regime and sufficient_vol:
            # Long entry: price breaks above Donchian upper channel
            if price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower channel
            elif price < donchian_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals