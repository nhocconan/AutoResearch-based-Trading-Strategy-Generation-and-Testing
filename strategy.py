#!/usr/bin/env python3
"""
Experiment #4655: 6h Camarilla Pivot Fade + Weekly Trend Filter + Volume Confirmation
HYPOTHESIS: Fade at 1d Camarilla H3/L3 (R3/S3) levels in ranging markets, filtered by weekly trend (price vs 20-week EMA) to avoid counter-trend trades. Volume confirmation (>1.5x MA20) ensures breakout validity. Works in bull (fade at support in uptrend) and bear (fade at resistance in downtrend). Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4655_6h_camarilla_weekly_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d for Camarilla, 1w for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Camarilla Pivot Levels (from prior 1d OHLC) ===
    if len(df_1d) >= 1:
        # Prior day's OHLC (shifted by 1 to avoid look-ahead)
        ph_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
        pl_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
        pc_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
        
        # Camarilla levels: based on prior day's range
        rng = ph_1d - pl_1d
        camarilla_h3 = pc_1d + 1.1 * rng / 4  # R3
        camarilla_l3 = pc_1d - 1.1 * rng / 4  # S3
    else:
        camarilla_h3 = camarilla_l3 = np.full(len(df_1d), np.nan)
    
    # === 1w Indicators: 20-period EMA for trend filter ===
    if len(df_1w) >= 20:
        ema_20w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    else:
        ema_20w = np.full(len(df_1w), np.nan)
    
    # Align HTF indicators to 6h timeframe
    if len(camarilla_h3) > 0:
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    else:
        camarilla_h3_aligned = np.full(n, np.nan)
        camarilla_l3_aligned = np.full(n, np.nan)
    
    if len(ema_20w) > 0:
        ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    else:
        ema_20w_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 14)  # Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_20w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation for breakouts/fades (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: weekly EMA20 direction
        weekly_uptrend = price > ema_20w_aligned[i]
        weekly_downtrend = price < ema_20w_aligned[i]
        
        # Fade conditions: price reaches Camarilla H3/L3 (R3/S3) with volume confirmation
        # In uptrend: fade at S3 (long) only
        # In downtrend: fade at R3 (short) only
        fade_long = (price <= camarilla_l3_aligned[i]) and vol_confirm and weekly_uptrend
        fade_short = (price >= camarilla_h3_aligned[i]) and vol_confirm and weekly_downtrend
        
        if fade_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif fade_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals