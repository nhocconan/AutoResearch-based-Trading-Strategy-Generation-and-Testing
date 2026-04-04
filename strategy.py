#!/usr/bin/env python3
"""
Experiment #4664: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: 1d price breaking Donchian(20) channels (from prior 20 daily bars) with volume confirmation captures momentum. 
1-week HMA(21) filter ensures trades align with higher-timeframe trend, reducing whipsaws in ranging markets. 
Works in bull (trend-following breakouts) and bear (avoids counter-trend entries via HMA filter).
Target: 7-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4664_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Donchian(20) from prior 20 days ===
    if len(close) >= 20:
        # Use prior 20 days' high/low (shifted by 1)
        ph = np.concatenate([[np.nan] * 20, high[:-20]])  # prior 20 days high
        pl = np.concatenate([[np.nan] * 20, low[:-20]])   # prior 20 days low
        
        # Rolling max/min of prior 20 days
        donchian_high = pd.Series(ph).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # === 1w Indicators: HMA(21) for trend filter ===
    if len(df_1w) >= 21:
        # Calculate HMA(21) on weekly close
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA of period half_len
        wma_half = pd.Series(df_1w['close'].values).ewm(span=half_len, adjust=False).mean().values
        # WMA of period full_len
        wma_full = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False).mean().values
        # Raw HMA = 2*WMA(half) - WMA(full)
        hma_raw = 2 * wma_half - wma_full
        # Final HMA = WMA of raw HMA with period sqrt_len
        hma_21 = pd.Series(hma_raw).ewm(span=sqrt_len, adjust=False).mean().values
    else:
        hma_21 = np.full(len(df_1w), np.nan)
    
    # Align HTF indicators to 1d timeframe
    if len(hma_21) > 0:
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 14)  # Donchian, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Trend filter: price above/below weekly HMA
        price_above_hma = price > hma_21_aligned[i]
        price_below_hma = price < hma_21_aligned[i]
        
        # Breakout conditions: price breaks Donchian high/low with volume confirmation AND trend filter
        breakout_long = price > donchian_high[i] and vol_breakout and price_above_hma
        breakout_short = price < donchian_low[i] and vol_breakout and price_below_hma
        
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