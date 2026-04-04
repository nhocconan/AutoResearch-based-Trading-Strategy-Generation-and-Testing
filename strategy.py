#!/usr/bin/env python3
"""
Experiment #4113: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 12h Hull Moving Average trend direction capture institutional order flow. Volume confirmation filters false breakouts. Works in bull/bear by using 12h trend as filter. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4113_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h HMA(21) for trend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 1:
        # Calculate Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        n_12h = len(close_12h)
        half = n_12h // 2
        sqrt_n = int(np.sqrt(n_12h))
        
        if n_12h >= 21 and half > 0 and sqrt_n > 0:
            wma_full = wma(close_12h, 21)
            wma_half = wma(close_12h, half)
            wma_sqrt = wma(close_12h, sqrt_n)
            
            # Pad to original length
            wma_full_padded = np.full(n_12h, np.nan)
            wma_half_padded = np.full(n_12h, np.nan)
            wma_sqrt_padded = np.full(n_12h, np.nan)
            
            if len(wma_full) > 0:
                wma_full_padded[20:] = wma_full
            if len(wma_half) > 0:
                wma_half_padded[half-1:] = wma_half
            if len(wma_sqrt) > 0:
                wma_sqrt_padded[sqrt_n-1:] = wma_sqrt
            
            hma_raw = 2 * wma_half_padded - wma_full_padded
            hma_12h = wma(hma_raw, sqrt_n)
            
            # Pad final HMA
            hma_final = np.full(n_12h, np.nan)
            if len(hma_12h) > 0:
                hma_final[sqrt_n-1:] = hma_12h
            
            # Align to 4h timeframe (shifted by 1 for completed bars only)
            hma_aligned = align_htf_to_ltf(prices, df_12h, hma_final)
        else:
            hma_aligned = np.full(n, np.nan)
    else:
        hma_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(20) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_aligned[i])):
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
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # 12h HMA trend filter: price above HMA = uptrend, below = downtrend
            uptrend = price > hma_aligned[i]
            downtrend = price < hma_aligned[i]
            
            # Long conditions: Donchian breakout up + uptrend + volume spike
            long_entry = breakout_up and uptrend
            
            # Short conditions: Donchian breakdown down + downtrend + volume spike
            short_entry = breakout_down and downtrend
            
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