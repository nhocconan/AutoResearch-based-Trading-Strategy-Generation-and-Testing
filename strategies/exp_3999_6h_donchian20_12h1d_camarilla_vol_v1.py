#!/usr/bin/env python3
"""
Experiment #3999: 6h Donchian(20) breakout + 12h/1d Camarilla pivot confluence + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h/1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) 
capture high-probability moves. Volume > 2.0x MA(50) confirms institutional participation. 
Uses discrete sizing (0.25) and ATR(20) trailing stop (2.0x) for risk control. 
Camarilla pivots derived from 1d timeframe provide institutional reference levels effective in both bull/bear regimes.
Target: 75-175 trades over 4 years (19-44/year). Works via confluence filtering reducing false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3999_6h_donchian20_12h1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivots ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        # Calculate Camarilla levels from previous 12h bar
        h_12h = df_12h['high'].values
        l_12h = df_12h['low'].values
        c_12h = df_12h['close'].values
        rng = h_12h - l_12h
        
        # Camarilla levels: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4)
        # S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
        r4_12h = c_12h + (rng * 1.1 / 2)
        r3_12h = c_12h + (rng * 1.1 / 4)
        s3_12h = c_12h - (rng * 1.1 / 4)
        s4_12h = c_12h - (rng * 1.1 / 2)
        
        # Align to 6h timeframe (shift by 1 for completed bar)
        r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
        r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
        s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
        s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    else:
        r4_12h_aligned = np.full(n, np.nan)
        r3_12h_aligned = np.full(n, np.nan)
        s3_12h_aligned = np.full(n, np.nan)
        s4_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for additional Camarilla confirmation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        h_1d = df_1d['high'].values
        l_1d = df_1d['low'].values
        c_1d = df_1d['close'].values
        rng_1d = h_1d - l_1d
        
        r4_1d = c_1d + (rng_1d * 1.1 / 2)
        r3_1d = c_1d + (rng_1d * 1.1 / 4)
        s3_1d = c_1d - (rng_1d * 1.1 / 4)
        s4_1d = c_1d - (rng_1d * 1.1 / 2)
        
        r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
        r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
        s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
        s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    else:
        r4_1d_aligned = np.full(n, np.nan)
        r3_1d_aligned = np.full(n, np.nan)
        s3_1d_aligned = np.full(n, np.nan)
        s4_1d_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(50) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[50:] = volume[50:] / vol_ma[50:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 50 + 10, 20 + 10)  # DC lookback, vol MA, ATR buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
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
        # Require strong volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine confluence from 12h and 1d Camarilla levels
            # Bullish confluence: price above R3 (both timeframes) AND breaking above R4
            # Bearish confluence: price below S3 (both timeframes) AND breaking below S4
            bullish_confluence_12h = price > r3_12h_aligned[i]
            bullish_confluence_1d = price > r3_1d_aligned[i]
            bearish_confluence_12h = price < s3_12h_aligned[i]
            bearish_confluence_1d = price < s3_1d_aligned[i]
            
            # Breakout conditions using Donchian
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Long: bullish confluence + breakout above Donchian high
            if bullish_confluence_12h and bullish_confluence_1d and breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: bearish confluence + breakout below Donchian low
            elif bearish_confluence_12h and bearish_confluence_1d and breakout_down:
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