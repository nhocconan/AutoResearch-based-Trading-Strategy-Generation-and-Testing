#!/usr/bin/env python3
"""
Experiment #5253: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts capture institutional momentum when confirmed by volume (>1.5x 20-bar average) and aligned with 12h HMA(21) trend. Exits via ATR(14) trailing stop (2.0x). Designed for 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by regime). Uses discrete position sizing (0.30) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5253_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 12h data for HMA(21) trend
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: HMA(21) for trend filter ===
    if len(df_12h) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 12 // 2
        sqrt_len = int(np.sqrt(12))
        
        def wma(arr, period):
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        if len(close_12h) >= 12:
            wma_half = wma(close_12h, half_len)
            wma_full = wma(close_12h, 12)
            wma_diff = 2 * wma_half - wma_full
            if len(wma_diff) >= sqrt_len:
                hma_12h = wma(wma_diff, sqrt_len)
                # Prepend NaNs for alignment
                hma_12h_full = np.full(len(close_12h), np.nan)
                hma_12h_full[half_len:half_len+len(hma_12h)] = hma_12h
                hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_full)
            else:
                hma_12h_aligned = np.full(n, np.nan)
        else:
            hma_12h_aligned = np.full(n, np.nan)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    donch_len = 20
    if n >= donch_len:
        # Upper channel: highest high over past 20 periods
        upper = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
        # Lower channel: lowest low over past 20 periods
        lower = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
    
    # === 4h Indicators: Volume confirmation (1.5x spike) ===
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
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: bullish if price > 12h HMA(21), bearish if price < 12h HMA(21)
        trend_bullish = price > hma_12h_aligned[i]
        trend_bearish = price < hma_12h_aligned[i]
        
        # Breakout conditions
        breakout_long = (price >= upper[i]) and trend_bullish and vol_confirm
        breakout_short = (price <= lower[i]) and trend_bearish and vol_confirm
        
        # Final entry conditions
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