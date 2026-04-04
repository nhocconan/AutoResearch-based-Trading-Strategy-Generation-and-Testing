#!/usr/bin/env python3
"""
Experiment #4120: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 1d HMA(21) trend direction capture strong momentum moves. 
Volume confirmation filters false breakouts. Works in bull markets via breakout continuation and in bear 
markets via mean reversion at channel edges. Uses discrete position sizing (0.25) to minimize fee drag. 
Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4120_4h_donchian20_1d_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d HMA(21) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        def wma(arr, window):
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21 = wma(wma_diff, sqrt_len)
        # Pad beginning with NaN
        hma_padded = np.full(len(close_1d), np.nan)
        hma_padded[half_len:] = hma_21[:len(close_1d)-half_len]
        hma_1d = hma_padded
        # Align to 4h timeframe (shifted by 1 for completed bars only)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
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
            np.isnan(hma_1d_aligned[i])):
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
            
            # HMA trend filter
            hma_bullish = price > hma_1d_aligned[i]
            hma_bearish = price < hma_1d_aligned[i]
            
            # Long conditions: breakout up + bullish trend OR mean reversion at lower band
            long_breakout = breakout_up and hma_bullish
            long_mean_rev = breakout_up and (price <= lowest_low[i-1] * 1.001) and hma_bearish  # slight bounce from lower band
            
            # Short conditions: breakout down + bearish trend OR mean reversion at upper band
            short_breakout = breakout_down and hma_bearish
            short_mean_rev = breakout_down and (price >= highest_high[i-1] * 0.999) and hma_bullish  # slight rejection from upper band
            
            if long_breakout or long_mean_rev:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_breakout or short_mean_rev:
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