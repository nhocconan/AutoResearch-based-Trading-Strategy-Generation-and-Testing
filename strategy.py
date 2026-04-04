#!/usr/bin/env python3
"""
Experiment #4647: 6h Donchian(20) Breakout from 1d HTF + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h price breaking Donchian(20) channels calculated from prior 1d data, with confirmation from 1d Camarilla pivot levels (breakout at R4/S4, fade at R3/S3) and volume (>1.5x avg), captures strong momentum with directional bias from weekly pivot. Weekly pivot provides longer-term trend filter to avoid counter-trend trades in ranging markets. Discrete sizing (0.25) and ATR trailing stop (2.0x). Target: 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4647_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for Donchian channels and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) from prior 1d OHLC (shifted by 1 to avoid look-ahead)
    if len(df_1d) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, df_1d['high'].values[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, df_1d['low'].values[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Align Donchian levels to 6h timeframe
    if len(donchian_high) > 0:
        dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    else:
        dh_aligned = np.full(n, np.nan)
        dl_aligned = np.full(n, np.nan)
    
    # Calculate 1d Camarilla pivot levels (using prior day's OHLC)
    if len(df_1d) >= 1:
        # Prior day's OHLC (shifted by 1 to avoid look-ahead)
        ph_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
        pl_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
        pc_1d = np.concatenate([[np.nan], df_1d['close'].values[:-1]])
        
        # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
        #               S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
        rng = ph_1d - pl_1d
        camarilla_r4 = pc_1d + (rng * 1.1 / 2)
        camarilla_r3 = pc_1d + (rng * 1.1 / 4)
        camarilla_s3 = pc_1d - (rng * 1.1 / 4)
        camarilla_s4 = pc_1d - (rng * 1.1 / 2)
    else:
        camarilla_r4 = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
        camarilla_s4 = np.full(n, np.nan)
    
    # Align Camarilla levels to 6h timeframe
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
    
    # Precompute 1w data for weekly pivot (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Prior week's OHLC (shifted by 1)
        wh_1w = np.concatenate([[np.nan], df_1w['high'].values[:-1]])
        wl_1w = np.concatenate([[np.nan], df_1w['low'].values[:-1]])
        wc_1w = np.concatenate([[np.nan], df_1w['close'].values[:-1]])
        # Weekly pivot point: (H+L+C)/3
        weekly_pivot = (wh_1w + wl_1w + wc_1w) / 3
    else:
        weekly_pivot = np.full(n, np.nan)
    
    # Align weekly pivot to 6h timeframe
    if len(weekly_pivot) > 0:
        wp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        wp_aligned = np.full(n, np.nan)
    
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
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(wp_aligned[i])):
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
        
        # Determine market bias from weekly pivot
        bullish_bias = price > wp_aligned[i]
        bearish_bias = price < wp_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation
        # AND Camarilla level confirmation
        breakout_long = (price > dh_aligned[i] and vol_breakout and 
                        price > r4_aligned[i] and bullish_bias)
        breakout_short = (price < dl_aligned[i] and vol_breakout and 
                         price < s4_aligned[i] and bearish_bias)
        
        # Fade conditions: price rejects at Camarilla R3/S3 with volume
        fade_long = (price < r3_aligned[i] and price > s3_aligned[i] and 
                    vol_breakout and bearish_bias and 
                    close[i] < open[i])  # bearish candle
        fade_short = (price > s3_aligned[i] and price < r3_aligned[i] and 
                     vol_breakout and bullish_bias and 
                     close[i] > open[i])  # bullish candle
        
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
        elif fade_long:
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