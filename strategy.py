#!/usr/bin/env python3
"""
Experiment #4849: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike (Revised)
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts in direction of 1d HMA21 trend with volume confirmation (>2x average) capture strong momentum moves. Uses ATR(14) stoploss (2.5x) to limit downside. Revised to increase trade frequency by: (1) Using 10-period Donchian for more frequent signals, (2) Reducing volume confirmation to 1.5x average, (3) Adding 4h EMA50 filter to avoid counter-trend whipsaws. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4849_4h_donchian10_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA21 for trend filter ===
    if len(df_1d) >= 21:
        # Hull Moving Average calculation
        half_len = len(df_1d) // 2
        sqrt_len = int(np.sqrt(len(df_1d)))
        
        # WMA function using pandas for efficiency
        def wma(values, window):
            return pd.Series(values).rolling(window=window, min_periods=window).mean().values
        
        close_1d = df_1d['close'].values
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, len(close_1d))
        wma_sqrt = wma(close_1d, sqrt_len)
        
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        hma_raw = 2 * wma_half - wma_full
        hma_1d = wma(hma_raw, sqrt_len)
    else:
        hma_1d = np.full(len(df_1d), np.nan)
    
    # Align HTF HMA21 to 4h timeframe
    if len(hma_1d) > 0:
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(10) channels (more frequent signals) ===
    high_roll = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_roll = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # === 4h Indicators: EMA50 for trend filter (avoid counter-trend) ===
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
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
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(10, 50, 20, 14)  # Donchian, EMA, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with trend alignment
        breakout_long = (price >= high_roll[i]) and (price > hma_1d_aligned[i]) and (price > ema_50[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < hma_1d_aligned[i]) and (price < ema_50[i]) and vol_confirm
        
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

</think>