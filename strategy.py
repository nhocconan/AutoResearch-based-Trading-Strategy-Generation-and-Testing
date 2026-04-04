#!/usr/bin/env python3
"""
Experiment #4350: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Daily Donchian breakouts capture institutional entry/exit points. Weekly HMA(21) filter ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts. Volume confirmation (>1.5x average) adds conviction. ATR-based trailing stop (2.5x) manages risk. Works in bull markets via upside breakouts with rising trend, and in bear markets via downside breakouts with falling trend. Targets 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4350_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1w HMA(21) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21): WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        # WMA helper
        def wma(arr, window):
            if len(arr) < window:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
        
        # Calculate WMAs
        wma_full = np.concatenate([np.full(20, np.nan), wma(close, 21)]) if len(close) >= 21 else np.full(len(close), np.nan)
        wma_half = np.concatenate([np.full(half-1, np.nan), wma(close, half)]) if len(close) >= half else np.full(len(close), np.nan)
        
        # 2*WMA(half) - WMA(full)
        diff = 2 * wma_half - wma_full
        
        # WMA of diff with sqrt(n) period
        hma_21 = np.concatenate([np.full(sqrt_n-1, np.nan), wma(diff, sqrt_n)]) if len(diff) >= sqrt_n else np.full(len(diff), np.nan)
        
        # Align to LTF
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channels (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
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
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: HMA slope direction
        hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
        hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # Long conditions: Price breaks above Donchian HIGH + volume + rising HMA
        long_entry = (price > donchian_high[i-1]) and volume_confirm and hma_rising
        
        # Short conditions: Price breaks below Donchian LOW + volume + falling HMA
        short_entry = (price < donchian_low[i-1]) and volume_confirm and hma_falling
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
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