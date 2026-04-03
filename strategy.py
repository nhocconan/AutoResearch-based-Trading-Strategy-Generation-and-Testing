#!/usr/bin/env python3
"""
Experiment #250: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Daily Donchian breakouts capture medium-term trends with clear structure. 
Weekly HMA(21) filters for higher timeframe trend alignment to avoid counter-trend trades. 
Volume confirmation ensures institutional participation. ATR-based stoploss manages risk.
Targets 7-25 trades/year on 1d timeframe (30-100 total over 4 years) to minimize fee drag.
Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        # Handle array length differences
        if len(wma_half) > 0 and len(wma_full) > 0:
            raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
            hma_1w = wma(raw_hma, sqrt_len)
            # Pad to original length
            hma_1w_full = np.full(len(close_1w), np.nan)
            hma_1w_full[-len(hma_1w):] = hma_1w
            hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_full)
        else:
            hma_1w_aligned = np.full(n, np.nan)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume SMA(20) for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    def atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        atr_vals = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        return atr_vals
    
    atr_14 = atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(vol_sma_20[i]) or np.isnan(atr_14[i]) or 
            np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian lower (trend change)
                if close[i] < lower_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks above Donchian upper (trend change)
                if close[i] > upper_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_sma_20[i]
        
        # Long: Price breaks above Donchian upper + above weekly HMA + volume
        if (close[i] > upper_20[i] and 
            close[i] > hma_1w_aligned[i] and 
            volume_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        
        # Short: Price breaks below Donchian lower + below weekly HMA + volume
        elif (close[i] < lower_20[i] and 
              close[i] < hma_1w_aligned[i] and 
              volume_confirmed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals