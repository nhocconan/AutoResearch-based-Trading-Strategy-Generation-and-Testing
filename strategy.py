#!/usr/bin/env python3
"""
Experiment #4353: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture institutional order flow when aligned with higher timeframe trend (12h HMA) and confirmed by volume spikes (>2.0x average). Works in bull via upside breakouts, in bear via downside breakouts. ATR trailing stop manages risk. Targets 75-200 total trades over 4 years (19-50/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4353_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
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
    
    # === Precompute HTF: 12h HMA(21) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Calculate HMA: WMA(2*WMA(n/2) - WMA(n)) with sqrt(n) period
        half_len = 12  # round(21/2)
        sqrt_len = 4   # round(sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        close_series = df_12h['close'].values
        wma_half = wma(close_series, half_len)
        wma_full = wma(close_series, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_12h = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_12h_padded = np.full(len(df_12h), np.nan)
        hma_12h_padded[half_len + sqrt_len - 1:] = hma_12h
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian(20) channels ===
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_12h_aligned[i])):
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter: price above/below 12h HMA
        price_above_hma = price > hma_12h_aligned[i]
        price_below_hma = price < hma_12h_aligned[i]
        
        if volume_confirm:
            # Long conditions: Donchian breakout above + price above 12h HMA
            long_entry = (price > donch_high[i-1]) and price_above_hma
            
            # Short conditions: Donchian breakout below + price below 12h HMA
            short_entry = (price < donch_low[i-1]) and price_below_hma
            
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
        else:
            signals[i] = 0.0
    
    return signals