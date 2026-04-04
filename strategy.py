#!/usr/bin/env python3
"""
Experiment #4694: 1h Donchian(20) Breakout + 4h EMA Trend + Volume Confirmation
HYPOTHESIS: 1h price breaking Donchian(20) channels with volume confirmation (>1.5x avg volume) and aligned with 4h EMA50 trend captures momentum while minimizing whipsaws. Uses 4h EMA for signal direction, 1h only for entry timing to avoid overtrading. Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year on 1h timeframe to avoid fee drag. Works in bull (breakouts with volume) and bear (short breakdowns with volume) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4694_1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute HTF: 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # === 4h Indicators: EMA50 for trend filter ===
    if len(df_4h) >= 50:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False).mean().values
    else:
        ema_4h = np.full(len(df_4h), np.nan)
    
    # Align HTF EMA50 to 1h timeframe
    if len(ema_4h) > 0:
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) from prior 20 bars ===
    # Use prior 20 bars' high/low (shifted by 1 to avoid look-ahead)
    ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 bars high
    pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 bars low
    
    # Rolling max/min of prior 20 bars
    donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, Volume MA, ATR, EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter ---
        if not in_session[i]:
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
        # Volume filter: confirmation for breakouts (>1.5x)
        vol_breakout = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i] and vol_breakout
        breakout_short = price < donchian_low[i] and vol_breakout
        
        # 4h EMA50 trend filter: only trade in direction of higher timeframe trend
        trend_filter_long = price > ema_4h_aligned[i]
        trend_filter_short = price < ema_4h_aligned[i]
        
        # Final entry conditions: breakout + volume + trend filter
        if breakout_long and trend_filter_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and trend_filter_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals