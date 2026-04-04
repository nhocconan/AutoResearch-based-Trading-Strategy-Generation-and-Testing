#!/usr/bin/env python3
"""
Experiment #4677: 4h Donchian(20) Breakout + 1d/1w Volume Spike + Chop Regime Filter
HYPOTHESIS: 4h price breaking Donchian(20) channels with volume confirmation (>2x average) 
in low chop regimes (CHOP < 38.2 = trending) captures momentum with low false signals. 
HTF: 1d Donchian for structure, 1w EMA for major trend filter. 
Target: 20-50 trades/year on 4h timeframe. Works in bull (breakouts) and bear (breakdowns) 
via symmetric long/short logic. Uses ATR(14) trailing stop (2.0x) for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4677_4h_donchian20_1d_1w_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian(20)
    df_1d = get_htf_data(prices, '1d')
    
    # Precompute HTF: 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Donchian(20) from prior 20 days ===
    if len(df_1d) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, df_1d['high'].values[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, df_1d['low'].values[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(len(df_1d), np.nan)
        donchian_low = np.full(len(df_1d), np.nan)
    
    # === 1w Indicators: EMA(50) for trend filter ===
    if len(df_1w) >= 50:
        ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_1w = np.full(len(df_1w), np.nan)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Choppiness Index (CHOP) regime filter ===
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index: higher = choppy, lower = trending"""
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        range_ = highest_high - lowest_low
        chop = 100 * np.log10(atr_sum / range_) / np.log10(period)
        # Prepend NaN for alignment
        return np.concatenate([[np.nan], chop])
    
    chop = calculate_chop(high, low, close, period=14)
    
    # Align HTF indicators to 4h timeframe
    if len(donchian_high) > 0:
        dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    else:
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
    
    if len(ema_1w) > 0:
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14, 14)  # Donchian, ATR, CHOP warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_1w_aligned[i])):
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
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        # Volume filter: confirmation for breakouts (>2.0x average)
        vol_breakout = vol_ratio[i] > 2.0
        
        # Trend filter: price above/below 1w EMA(50)
        uptrend = price > ema_1w_aligned[i]
        downtrend = price < ema_1w_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation
        breakout_long = price > dh_aligned[i] and vol_breakout and trending_regime and uptrend
        breakout_short = price < dl_aligned[i] and vol_breakout and trending_regime and downtrend
        
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