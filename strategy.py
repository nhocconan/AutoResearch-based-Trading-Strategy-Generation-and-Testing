#!/usr/bin/env python3
"""
Experiment #6409: 4h Donchian(20) breakout + 1d ATR volatility filter + volume confirmation
HYPOTHESIS: 4h Donchian breakouts with volume confirmation (>2.0x 20-period average) and 1d ATR-based volatility filter (ATR ratio > 1.5) capture institutional breakouts with reduced false signals. The volatility filter ensures we only trade during expanded volatility regimes, reducing whipsaws in ranging markets. Discrete sizing (0.25) balances profit potential and drawdown control. Target: 75-200 trades over 4 years. Works in bull via upside breakouts with volume, in bear via downside breakouts with volume, and avoids ranging markets via volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6409_4h_donchian20_1d_atr_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ATR volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 14:
        # Calculate True Range and ATR(14) on daily timeframe
        tr1 = df_1d['high'] - df_1d['low']
        tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
        tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        # Calculate ATR ratio: current ATR / 20-period average ATR
        atr_avg_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
        atr_ratio = atr_1d / np.where(atr_avg_20 > 0, atr_avg_20, 1)
        # Align to 4h timeframe (shifted by 1 day for lookback safety)
        atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    else:
        atr_ratio_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 20) + 1  # Donchian, volume avg, ATR, HTF ATR ratio lookback + 1
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(atr_ratio_aligned[i])):
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
                # 3. Volatility contraction (ATR ratio < 1.0) - exit ranging markets
                if price <= stop_price or price <= donchian_low[i] or atr_ratio_aligned[i] < 1.0:
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
                # 3. Volatility contraction (ATR ratio < 1.0) - exit ranging markets
                if price >= stop_price or price >= donchian_high[i] or atr_ratio_aligned[i] < 1.0:
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
        volatility_expanded = atr_ratio_aligned[i] > 1.5  # Volatility filter: trade only when ATR > 1.5x 20-day average
        
        # Entry logic: Donchian breakout with volume and volatility confirmation
        if breakout_up and volume_confirmed and volatility_expanded:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and volatility_expanded:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals