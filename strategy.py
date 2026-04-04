#!/usr/bin/env python3
"""
Experiment #4614: 1h Camarilla Pivot Breakout with 4h/1d HTF Filter
HYPOTHESIS: 1h price breaking Camarilla R4/S4 levels (from prior 1d) with volume confirmation and aligned with 4h/1d trend captures strong momentum breakouts. Uses 4h/1d for signal direction (only trade long when 4h/1d both bullish, short when both bearish) and 1h only for entry timing precision. Session filter (08-20 UTC) reduces noise. Discrete sizing (0.20) and ATR trailing stop manage risk. Target: 60-150 total trades over 4 years = 15-37/year on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4614_1h_camarilla_pivot_4h_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours ONCE (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # Precompute HTF: 4h and 1d data for trend and Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA(21) for trend direction ===
    if len(df_4h) >= 1:
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    else:
        ema_4h = np.array([])
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h) if len(ema_4h) > 0 else np.full(n, np.nan)
    
    # === 1d Indicators: EMA(50) for HTF trend filter ===
    if len(df_1d) >= 1:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_1d = np.array([])
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d) if len(ema_1d) > 0 else np.full(n, np.nan)
    
    # === 1d Indicators: Camarilla pivot levels from prior 1d OHLC ===
    if len(df_1d) >= 1:
        # Use prior day's OHLC (shifted by 1 to avoid look-ahead)
        ph = np.concatenate([[np.nan], df_1d['high'].values[:-1]])  # prior day high
        pl = np.concatenate([[np.nan], df_1d['low'].values[:-1]])   # prior day low
        pc = np.concatenate([[np.nan], df_1d['close'].values[:-1]]) # prior day close
        
        # Camarilla calculations
        rng = ph - pl
        camarilla_r4 = pc + (rng * 1.1 / 2.0)
        camarilla_r3 = pc + (rng * 1.1 / 4.0)
        camarilla_s3 = pc - (rng * 1.1 / 4.0)
        camarilla_s4 = pc - (rng * 1.1 / 2.0)
    else:
        camarilla_r4 = np.array([])
        camarilla_r3 = np.array([])
        camarilla_s3 = np.array([])
        camarilla_s4 = np.array([])
    
    # Align Camarilla levels to 1h timeframe
    if len(camarilla_r4) > 0:
        r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        r4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 21, 50)  # Volume MA, 4h EMA, 1d EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Determine HTF Trend Direction (4h & 1d both bullish/bearish) ---
        # For long: need both 4h and 1d bullish (price above their EMAs)
        htf_bullish = (price > ema_4h_aligned[i]) and (price > ema_1d_aligned[i])
        # For short: need both 4h and 1d bearish (price below their EMAs)
        htf_bearish = (price < ema_4h_aligned[i]) and (price < ema_1d_aligned[i])
        
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
        # Volume filter: confirmation for breakouts (>1.5x avg volume)
        vol_breakout = vol_ratio[i] > 1.5
        
        # Breakout conditions: price breaks R4/S4 with volume confirmation AND HTF alignment
        breakout_long = price > r4_aligned[i] and vol_breakout and htf_bullish
        breakout_short = price < s4_aligned[i] and vol_breakout and htf_bearish
        
        # Enter long on bullish breakout with HTF alignment
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        # Enter short on bearish breakout with HTF alignment
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