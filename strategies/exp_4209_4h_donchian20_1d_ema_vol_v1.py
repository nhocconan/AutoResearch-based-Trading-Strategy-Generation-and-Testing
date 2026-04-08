#!/usr/bin/env python3
"""
Experiment #4209: 4h Donchian(20) breakout + 1d/1w HTF trend + volume confirmation
HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture momentum when aligned with higher timeframe trends (1d/1w EMA cross) and confirmed by volume (>1.6x average). 
Uses discrete position sizing (0.25) to limit fee churn, targeting 75-200 total trades over 4 years (19-50/year). 
Works in both bull and bear markets by requiring HTF trend alignment (avoids counter-trend trades) and volume confirmation (filters false breakouts).
ATR-based trailing stop (2.5x) manages risk. Proven pattern from DB top performers.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4209_4h_donchian20_1d_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 1d and 1w EMA for trend alignment ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA cross (50/200) for trend filter
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
        uptrend_1d = ema_50_1d_aligned > ema_200_1d_aligned
    else:
        uptrend_1d = np.full(n, False)
    
    # 1w EMA cross (20/50) for longer-term trend
    if len(df_1w) >= 20:
        close_1w = df_1w['close'].values
        ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
        uptrend_1w = ema_20_1w_aligned > ema_50_1w_aligned
    else:
        uptrend_1w = np.full(n, False)
    
    # Combined HTF trend: both timeframes must agree (more selective)
    htf_uptrend = uptrend_1d & uptrend_1w
    htf_downtrend = (~uptrend_1d) & (~uptrend_1w)
    
    # === 4h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 200)  # Donchian, vol MA, ATR, 1d EMA200
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) if 'ema_50_1d_aligned' in locals() else False or
            np.isnan(ema_200_1d_aligned[i]) if 'ema_200_1d_aligned' in locals() else False or
            np.isnan(ema_20_1w_aligned[i]) if 'ema_20_1w_aligned' in locals() else False or
            np.isnan(ema_50_1w_aligned[i]) if 'ema_50_1w_aligned' in locals() else False):
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
        # Require volume confirmation (> 1.6x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.6
        
        if volume_confirm:
            # Donchian breakout conditions
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # Long conditions: breakout up + HTF uptrend
            long_entry = breakout_up and htf_uptrend[i]
            
            # Short conditions: breakout down + HTF downtrend
            short_entry = breakout_dn and htf_downtrend[i]
            
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