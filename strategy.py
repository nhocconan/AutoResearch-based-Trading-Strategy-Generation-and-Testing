#!/usr/bin/env python3
"""
Experiment #090: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Daily Donchian channel breakouts aligned with weekly Hull Moving Average trend direction,
confirmed by daily volume spikes, provide high-probability entries in both bull and bear markets.
The 1d timeframe minimizes fee drag while capturing multi-week trends. Weekly HMA acts as a robust
trend filter that adapts quickly to regime changes. Volume confirmation ensures institutional
participation. Targets 7-25 trades/year (30-100 total over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_htf_hma_vol_v1"
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
    
    # Calculate HMA(21) on weekly close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = len(close_1w) // 2
        sqrt_n = int(np.sqrt(len(close_1w)))
        if half > 0 and sqrt_n > 0:
            wma_half = pd.Series(close_1w).ewm(span=half*2, adjust=False).mean()
            wma_full = pd.Series(close_1w).ewm(span=len(close_1w), adjust=False).mean()
            raw_hma = 2 * wma_half - wma_full
            hma_21 = raw_hma.ewm(span=sqrt_n, adjust=False).mean().values
            hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
        else:
            hma_21_aligned = np.full(n, np.nan)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, 1.0)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_price_since_entry = 0.0
    min_price_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            # Update max/min since entry
            if position_side > 0:  # Long
                max_price_since_entry = max(max_price_since_entry, high[i])
                # Stoploss: 2.5 * ATR below entry OR price < weekly HMA
                stop_level = entry_price - 2.5 * atr[i]
                trend_stop = hma_21_aligned[i]
                if low[i] < stop_level or close[i] < trend_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    max_price_since_entry = 0.0
                    min_price_since_entry = 0.0
                    continue
                # Take profit at 3 * ATR profit
                if high[i] >= entry_price + 3.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    max_price_since_entry = 0.0
                    min_price_since_entry = 0.0
                    continue
            else:  # Short
                min_price_since_entry = min(min_price_since_entry, low[i])
                # Stoploss: 2.5 * ATR above entry OR price > weekly HMA
                stop_level = entry_price + 2.5 * atr[i]
                trend_stop = hma_21_aligned[i]
                if high[i] > stop_level or close[i] > trend_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    max_price_since_entry = 0.0
                    min_price_since_entry = 0.0
                    continue
                # Take profit at 3 * ATR profit
                if low[i] <= entry_price - 3.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    max_price_since_entry = 0.0
                    min_price_since_entry = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high + volume spike + price > weekly HMA
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper band
            vol_ratio[i] > 1.8 and           # Volume spike
            close[i] > hma_21_aligned[i]     # Above weekly HMA (uptrend)
        )
        
        # Short: Price breaks below Donchian low + volume spike + price < weekly HMA
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below lower band
            vol_ratio[i] > 1.8 and           # Volume spike
            close[i] < hma_21_aligned[i]     # Below weekly HMA (downtrend)
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            max_price_since_entry = high[i]
            min_price_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            max_price_since_entry = high[i]
            min_price_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals