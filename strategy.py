#!/usr/bin/env python3
"""
Experiment #4679: 6h Donchian(20) Breakout + 12h Volume Spike + 1d ADX Trend Filter
HYPOTHESIS: 6h price breaking Donchian(20) channels with volume confirmation (>2x MA) and 1d ADX>25 (trending) captures strong momentum breakouts.
Works in bull (breakouts long) and bear (breakdowns short). Volume spike filters false breakouts, ADX ensures trending environment.
Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4679_6h_donchian20_12h_vol_1d_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    # Precompute HTF: 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    if len(df_12h) >= 20:
        vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    else:
        vol_ma_12h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: ADX(14) for trend strength ===
    if len(df_1d) >= 14:
        # Calculate +DM, -DM, TR
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr_1d = pd.Series(tr).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / np.where(atr_1d == 0, 1, atr_1d)
        minus_di = 100 * minus_dm_smooth / np.where(atr_1d == 0, 1, atr_1d)
        
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
        adx = pd.Series(dx).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
        
        # Prepend NaN for first element (since calculations started from index 1)
        adx = np.concatenate([[np.nan], adx[:-1]])
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) from prior 20 bars ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
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
    
    warmup = max(20, 20)  # Donchian, Volume MA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_aligned[i])):
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
        # Volume filter: confirmation for breakouts (>2x MA)
        vol_breakout = vol_ratio[i] > 2.0
        # 12h volume spike: current volume > 1.5x 12h MA
        vol_spike_12h = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        # ADX filter: trending market (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation
        breakout_long = price > donchian_high[i] and vol_breakout and vol_spike_12h and strong_trend
        breakout_short = price < donchian_low[i] and vol_breakout and vol_spike_12h and strong_trend
        
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