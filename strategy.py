#!/usr/bin/env python3
"""
Experiment #4618: 1d Donchian(20) Breakout + HMA Trend + Volume Confirmation
HYPOTHESIS: 1d price breaking Donchian(20) channels with HMA(21) trend filter and volume confirmation (>1.3x average) captures strong momentum moves in both bull and bear markets. Uses 1w HTF for HMA trend alignment to avoid look-ahead. Discrete sizing (0.25) and ATR(14) trailing stop (2.5x) manage risk. Target: 7-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4618_1d_donchian20_hma_vol_v1"
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
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 1:
        close_1w = df_1w['close'].values.astype(np.float64)
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = max(1, len(close_1w) // 2)
        sqrt_len = max(1, int(np.sqrt(len(close_1w))))
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, len(close_1w))
        wma_diff = 2 * wma_half - wma_full
        hma_1w = wma(wma_diff, sqrt_len)
        # Pad to original length
        hma_1w_padded = np.full(len(close_1w), np.nan)
        hma_1w_padded[half_len - 1:half_len - 1 + len(hma_1w)] = hma_1w
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
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
    
    warmup = max(20, 20)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(hma_1w_aligned[i]) or
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
        # Volume filter: confirmation (>1.3x average volume)
        vol_confirm = vol_ratio[i] > 1.3
        
        # Trend filter: HMA(21) direction from 1w
        hma_up = hma_1w_aligned[i] > hma_1w_aligned[i-1] if i > 0 else False
        hma_down = hma_1w_aligned[i] < hma_1w_aligned[i-1] if i > 0 else False
        
        # Breakout conditions: price breaks Donchian channels with volume and trend confirmation
        breakout_long = price > donchian_high[i] and vol_confirm and hma_up
        breakout_short = price < donchian_low[i] and vol_confirm and hma_down
        
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