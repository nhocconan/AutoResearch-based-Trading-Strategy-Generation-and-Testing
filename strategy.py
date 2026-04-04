#!/usr/bin/env python3
"""
Experiment #4674: 1h Donchian(20) Breakout + 4h EMA Trend + 1d Volume Spike + Session Filter
HYPOTHESIS: 1h price breaking Donchian(20) channels with volume confirmation from 1d and trend filter from 4h EMA captures momentum while minimizing false breakouts. Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4674_1h_donchian20_4h_ema_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    # Precompute HTF: 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA(21) for trend filter ===
    if len(df_4h) >= 21:
        close_4h = df_4h['close'].values
        ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian(20) breakout ===
    donchian_len = 20
    if n >= donchian_len:
        donchian_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().shift(1).values
        donchian_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().shift(1).values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Precompute session hours (08-20 UTC)
    session_hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(donchian_len, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = session_hours[i]
        in_session = (8 <= hour <= 20)  # UTC 08-20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
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
        # Volume filter: confirmation for breakouts (>1.5x 1d average)
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        vol_breakout = vol_ratio > 1.5
        
        # Trend filter: 4h EMA direction (price above/below EMA)
        price_above_ema = price > ema_4h_aligned[i]
        price_below_ema = price < ema_4h_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume and trend confirmation
        breakout_long = price > donchian_high[i] and vol_breakout and price_above_ema
        breakout_short = price < donchian_low[i] and vol_breakout and price_below_ema
        
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