#!/usr/bin/env python3
"""
Experiment #4343: 4h Donchian(20) + 12h HMA Trend + Volume Spike + ATR Stop
HYPOTHESIS: Donchian(20) breakout on 4h captures structural moves when aligned with 12h HMA(21) trend and confirmed by volume (>2.0x MA20). Works in bull via upside breakouts, in bear via downside breakdowns. Uses 12h timeframe for HTF trend filter to reduce whipsaw. Targets 75-200 total trades over 4 years (19-50/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4343_4h_donchian20_12h_hma_vol_v1"
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
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        close_12h = df_12h['close'].values
        n_12h = len(close_12h)
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        wma_half = wma(close_12h, half)
        wma_full = wma(close_12h, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        
        # Pad to align
        hma_raw = np.full(n_12h, np.nan)
        start_idx = 21 - 1
        if len(wma_2x_sub) > 0:
            end_idx = start_idx + len(wma_2x_sub)
            if end_idx <= n_12h:
                hma_raw[start_idx:end_idx] = wma(wma_2x_sub, sqrt_n)
        
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_raw)
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
        
        # Donchian breakout conditions
        breakout_up = high[i] > donch_high[i-1]  # New high above prior channel
        breakout_down = low[i] < donch_low[i-1]  # New low below prior channel
        
        if volume_confirm:
            # Long conditions: Upside breakout + price above 12h HMA
            long_entry = breakout_up and price_above_hma
            
            # Short conditions: Downside breakout + price below 12h HMA
            short_entry = breakout_down and price_below_hma
            
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