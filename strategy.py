#!/usr/bin/env python3
"""
Experiment #5125: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts aligned with 1d HMA(21) trend capture strong momentum with lower trade frequency. 
Volume > 1.8x average confirms participation. ATR(14) trailing stop (2.5x) manages risk. 
Designed for 12-37 trades/year on 12h timeframe to minimize fee drag. Works in bull markets (breakouts with trend) 
and bear markets (breakdowns with trend). Uses discrete position sizing (0.30) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5125_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for HMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA(21) for trend ===
    if len(df_1d) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_1d = df_1d['close'].values
        n_1d = len(close_1d)
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        wma_half = wma(close_1d, half_n)
        wma_full = wma(close_1d, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_1d = wma(wma_diff, sqrt_n)
        
        # Pad to match original length
        hma_1d_padded = np.full(n_1d, np.nan)
        hma_1d_padded[half_n:half_n + len(hma_1d)] = hma_1d
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_padded)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation (1.8x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.8x)
        vol_confirm = vol_ratio[i] > 1.8
        
        # Donchian breakout conditions with 1d HMA trend filter
        # Long: Donchian breakout above + price > 1d HMA (uptrend)
        # Short: Donchian breakdown below + price < 1d HMA (downtrend)
        breakout_long = (price >= high_roll[i]) and (price > hma_1d_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_1d_aligned[i]) and vol_confirm
        
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