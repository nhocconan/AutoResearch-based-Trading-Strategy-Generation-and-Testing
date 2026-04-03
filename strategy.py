#!/usr/bin/env python3
"""
Experiment #122: 12h Donchian Breakout + Volume Spike + 1d Trend Filter

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 12h volume spike (>2.0x average) 
and aligned with 1d trend (price > EMA50 for longs, < EMA50 for shorts), captures high-probability 
trend continuation moves. Uses discrete position sizing (0.30) and ATR-based stoploss (2.5x ATR) 
to manage risk. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize 
fee drag while working in both bull and bear markets via trend filter.
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
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators (Pre-compute for efficiency) ===
    # Donchian channels (20-period) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        vol_12h = df_12h['volume'].values
        
        # Upper and lower Donchian channels
        donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
        
        # Volume ratio (current vs 20-period average)
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        
        # Align to 12h timeframe
        donch_high_12h = align_htf_to_ltf(prices, df_12h, donch_high)
        donch_low_12h = align_htf_to_ltf(prices, df_12h, donch_low)
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        donch_high_12h = np.full(n, np.nan)
        donch_low_12h = np.full(n, np.nan)
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets ---
        price_above_1d_ema = close[i] > ema_50_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
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
                # Take profit at Donchian lower channel (trailing exit)
                if close[i] <= donch_low_12h[i]:
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
                # Take profit at Donchian upper channel (trailing exit)
                if close[i] >= donch_high_12h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper channel with volume and trend alignment
        long_condition = (
            close[i] > donch_high_12h[i] and 
            volume_spike and 
            price_above_1d_ema
        )
        
        # Short: Price breaks below Donchian lower channel with volume and trend alignment
        short_condition = (
            close[i] < donch_low_12h[i] and 
            volume_spike and 
            price_below_1d_ema
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = entry_price
            lowest_since_entry = entry_price
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = entry_price
            lowest_since_entry = entry_price
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals