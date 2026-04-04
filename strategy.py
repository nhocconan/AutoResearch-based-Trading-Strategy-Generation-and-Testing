#!/usr/bin/env python3
"""
Experiment #5038: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 1d timeframe, Donchian(20) breakouts aligned with weekly HMA trend capture strong momentum with low frequency. Weekly HMA acts as trend filter: only long when price > weekly HMA(21), short when price < weekly HMA(21). Volume > 1.5x average confirms participation. ATR(14) trailing stop (2.0x) manages risk. Designed for 7-25 trades/year on 1d timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5038_1d_donchian20_1w_hma_vol_v1"
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
    
    # === 1w Indicators: HMA(21) for trend filter ===
    if len(df_1w) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_series = pd.Series(df_1w['close'].values)
        half_len = len(close_series) // 2
        sqrt_len = int(np.sqrt(len(close_series)))
        
        wma_half = wma(close_series.values, half_len) if half_len > 0 else np.full_like(close_series.values, np.nan)
        wma_full = wma(close_series.values, len(close_series)) if len(close_series) > 0 else np.full_like(close_series.values, np.nan)
        
        if len(wma_half) > 0 and len(wma_full) > 0:
            # Pad arrays to match original length
            wma_half_padded = np.full(len(close_series), np.nan)
            wma_full_padded = np.full(len(close_series), np.nan)
            wma_half_padded[-len(wma_half):] = wma_half
            wma_full_padded[-len(wma_full):] = wma_full
            
            raw_hma = 2 * wma_half_padded - wma_full_padded
            hma_values = wma(raw_hma, sqrt_len) if sqrt_len > 0 else np.full_like(raw_hma, np.nan)
            
            # Pad hma_values
            hma_padded = np.full(len(close_series), np.nan)
            if len(hma_values) > 0:
                hma_padded[-len(hma_values):] = hma_values
            hma_1w = hma_padded
        else:
            hma_1w = np.full(len(close_series), np.nan)
    else:
        hma_1w = np.full(len(df_1w), np.nan)
    
    # Align HMA to 1d timeframe
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume confirmation (1.5x spike) ===
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
        
        # Trend filter: price vs weekly HMA
        trend_long = price > hma_1w_aligned[i]
        trend_short = price < hma_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = (price >= high_roll[i]) and trend_long and vol_confirm
        breakout_short = (price <= low_roll[i]) and trend_short and vol_confirm
        
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