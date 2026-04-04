#!/usr/bin/env python3
"""
Experiment #4652: 12h Donchian Breakout with 1d/1w HTF Trend Filter and Volume Confirmation
HYPOTHESIS: 12h price breaking Donchian(20) channels with 1d EMA50 trend alignment and 1w volume confirmation captures sustained moves in both bull and bear markets. Uses discrete sizing (0.25) and ATR trailing stop (2.5x). Target: 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4652_12h_donchian20_1d_ema_1w_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    # Precompute HTF: 1w data for volume MA confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA50 for trend filter
    if len(df_1d) >= 1:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # Calculate 1w volume MA20 for confirmation
    if len(df_1w) >= 1:
        vol_ma_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    else:
        vol_ma_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channels (20) ===
    def donchian_channel(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume ratio for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14, 50)  # Donchian, ATR, EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume confirmation: current volume > 1.5x 12h MA AND > 1.2x 1w MA
        vol_confirm_12h = vol_ratio[i] > 1.5
        vol_confirm_1w = volume[i] > 1.2 * vol_ma_1w_aligned[i]
        vol_confirm = vol_confirm_12h and vol_confirm_1w
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema_1d_aligned[i]
        downtrend = price < ema_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = price > upper_20[i] and vol_confirm and uptrend
        breakout_short = price < lower_20[i] and vol_confirm and downtrend
        
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