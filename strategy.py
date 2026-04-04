#!/usr/bin/env python3
"""
Experiment #5110: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 1d timeframe, Donchian(20) breakouts aligned with 1w HMA(21) trend capture strong momentum across bull and bear markets. 
Volume > 1.8x average confirms institutional participation. ATR(14) trailing stop (2.5x) manages risk during volatile periods. 
Designed for 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to minimize fee drag. 
Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5110_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for HMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: HMA(21) for trend ===
    if len(df_1w) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        n_1w = len(close_1w)
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        wma_half = wma(close_1w, half_n)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_1w = wma(wma_diff, sqrt_n)
        
        # Pad to match original length
        hma_1w_padded = np.full(n_1w, np.nan)
        hma_1w_padded[half_n:half_n + len(hma_1w)] = hma_1w
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation (1.8x spike) ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        
        # Donchian breakout conditions with 1w HMA trend filter
        # Long: Donchian breakout above + price > 1w HMA (uptrend)
        # Short: Donchian breakdown below + price < 1w HMA (downtrend)
        breakout_long = (price >= high_roll[i]) and (price > hma_1w_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_1w_aligned[i]) and vol_confirm
        
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