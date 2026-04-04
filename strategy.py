#!/usr/bin/env python3
"""
Experiment #5259: 6h Donchian(20) Breakout + 12h ATR Regime + Volume Spike
HYPOTHESIS: On 6h timeframe, price breaking Donchian(20) channels from prior completed 6h bar with volume spike (>1.8x) in the direction of 12h ATR-based regime (trending if ATR(14) > ATR(50), ranging otherwise) captures institutional breakouts while filtering chop. Uses discrete position sizing (0.25) to balance return and drawdown. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Works in bull markets (breakouts continue uptrend) and bear markets (breakouts continue downtrend) by aligning with higher timeframe volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5259_6h_donchian20_12h_atr_regime_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 6h data for Donchian channels (structure) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) >= 20:
        # Donchian(20) on prior completed 6h bar (shift(1) in align)
        donch_high = pd.Series(df_6h['high']).rolling(window=20, min_periods=20).max().shift(1).values
        donch_low = pd.Series(df_6h['low']).rolling(window=20, min_periods=20).min().shift(1).values
        donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
        donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    else:
        donch_high_aligned = np.full(n, np.nan)
        donch_low_aligned = np.full(n, np.nan)
    
    # === HTF: 12h data for ATR regime filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 50:
        close_12h = pd.Series(df_12h['close'].values)
        high_12h = pd.Series(df_12h['high'].values)
        low_12h = pd.Series(df_12h['low'].values)
        # True Range
        tr1 = high_12h[1:] - low_12h[1:]
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
        atr_50 = pd.Series(tr_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        # Regime: 1 = trending (ATR14 > ATR50), 0 = ranging (ATR14 <= ATR50)
        regime_trending = (atr_14 > atr_50).astype(float)
        regime_trending_aligned = align_htf_to_ltf(prices, df_12h, regime_trending)
    else:
        regime_trending_aligned = np.zeros(n)
    
    # === 6h Indicators: Volume confirmation (1.8x spike) ===
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
    
    warmup = max(20, 20, 50, 20, 14)  # Donchian, Vol MA, ATR regime, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(regime_trending_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.8x)
        vol_confirm = vol_ratio[i] > 1.8
        
        # Regime filter: only trade in trending markets (ATR14 > ATR50 on 12h)
        regime_filter = regime_trending_aligned[i] > 0.5
        
        # Donchian breakout in trending regime
        breakout_long = (price >= donch_high_aligned[i]) and vol_confirm and regime_filter
        breakout_short = (price <= donch_low_aligned[i]) and vol_confirm and regime_filter
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals