#!/usr/bin/env python3
"""
Experiment #125: 12h Donchian Breakout + 1d Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe capture medium-term trends, confirmed by 1d volume spikes (>2x average) to ensure institutional participation. ATR-based stoploss (2.5x) manages risk. This strategy targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while participating in both bull and bear markets. The 12h timeframe reduces noise and overtrading common in lower timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 12h Indicators ===
    # Calculate Donchian(20) channels on 12h using historical data
    # We need to map each 12h bar to the prior completed 12h bar's Donchian
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_base = np.full(n, np.nan)  # Midpoint for filtering
    
    # Pre-compute 12h OHLC arrays for Donchian calculation
    # Create a DataFrame with 12h-aligned timestamps for rolling calculation
    # We'll use the prices data but calculate Donchian on 12h bars
    # Approach: calculate on 12h data then align back
    
    # Get 12h data for indicator calculation (separate from volume HTF)
    df_12h_ind = get_htf_data(prices, '12h')
    if len(df_12h_ind) >= 20:
        high_12h = df_12h_ind['high'].values
        low_12h = df_12h_ind['low'].values
        # Donchian(20): highest high and lowest low of past 20 periods
        dh_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        dl_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        db_20 = (dh_20 + dl_20) / 2.0  # Midpoint
        # Align back to 12h timeframe
        donchian_high_aligned = align_htf_to_ltf(prices, df_12h_ind, dh_20)
        donchian_low_aligned = align_htf_to_ltf(prices, df_12h_ind, dl_20)
        donchian_base_aligned = align_htf_to_ltf(prices, df_12h_ind, db_20)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
        donchian_base_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_base_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using available data up to i
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian(20) high with volume confirmation
        long_condition = (
            close[i] > donchian_high_aligned[i] and  # Breakout above upper band
            vol_ratio_1d_aligned[i] > 2.0            # Volume spike (>2x average)
        )
        
        # Short: Price breaks below Donchian(20) low with volume confirmation
        short_condition = (
            close[i] < donchian_low_aligned[i] and   # Breakdown below lower band
            vol_ratio_1d_aligned[i] > 2.0            # Volume spike (>2x average)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals