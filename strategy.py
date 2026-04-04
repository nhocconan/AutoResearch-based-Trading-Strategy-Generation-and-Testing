#!/usr/bin/env python3
"""
Experiment #4624: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: 1d price breaking 20-day Donchian channels with 1-week HMA trend filter and volume confirmation captures sustained momentum moves. Uses discrete sizing (0.25) and ATR trailing stop (2.5x) for risk management. Target: 20-50 trades over 4 years (5-12/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4624_1d_donchian20_1w_hma_vol_v1"
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
    
    # Calculate HMA(21) on weekly close
    if len(df_1w) >= 1:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = len(df_1w) // 2
        sqrt_len = int(np.sqrt(len(df_1w)))
        if half_len > 0 and sqrt_len > 0:
            wma_half = pd.Series(df_1w['close'].values).ewm(span=half_len, adjust=False).mean()
            wma_full = pd.Series(df_1w['close'].values).ewm(span=len(df_1w), adjust=False).mean()
            raw_hma = 2 * wma_half - wma_full
            hma_21 = pd.Series(raw_hma.values).ewm(span=sqrt_len, adjust=False).mean().values
        else:
            hma_21 = np.full(len(df_1w), np.nan)
    else:
        hma_21 = np.array([])
    
    # Align HMA to daily timeframe
    if len(hma_21) > 0:
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === Daily Indicators: Donchian(20) channels ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Daily Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Daily Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
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
        # Volume confirmation: >1.3x average volume
        vol_confirm = vol_ratio[i] > 1.3
        
        # Trend filter: price above/below weekly HMA
        uptrend = price > hma_21_aligned[i]
        downtrend = price < hma_21_aligned[i]
        
        # Donchian breakout conditions with volume and trend confirmation
        breakout_long = price > donchian_high[i] and vol_confirm and uptrend
        breakout_short = price < donchian_low[i] and vol_confirm and downtrend
        
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