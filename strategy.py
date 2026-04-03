#!/usr/bin/env python3
"""
Experiment #404: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation

HYPOTHESIS: Daily Donchian channel breakouts (20-period) aligned with weekly HMA(21) trend and 
volume confirmation (>1.5x average) captures strong momentum moves in both bull and bear markets. 
The weekly trend filter ensures we only trade in the direction of the higher timeframe trend, 
reducing whipsaw. Volume confirmation ensures institutional participation. Targets 7-25 trades/year 
on 1d timeframe (30-100 total over 4 years) to minimize fee drag while maintaining statistical 
significance. Uses discrete position sizing (0.25) and ATR-based stoploss (2.5x) to control risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_1w_hma_vol_v1"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if len(high) >= 20:
        donchian_high[20:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[20:]
        donchian_low[20:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[20:]
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, 1.0)
    if len(volume) >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or Donchian opposite) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                tr[0] = high[0] - low[0]
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_14 = 0.0
            
            if position_side > 0:  # Long position
                # Update highest close since entry
                highest_since_entry = max(highest_since_entry, close[i])
                stop_level = highest_since_entry - 2.5 * atr_14
                # Exit if stoploss hit OR price closes below Donchian low (trend reversal)
                if low[i] < stop_level or close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    highest_since_entry = 0.0
                    continue
            else:  # Short position
                # Update lowest close since entry
                lowest_since_entry = min(lowest_since_entry, close[i])
                stop_level = lowest_since_entry + 2.5 * atr_14
                # Exit if stoploss hit OR price closes above Donchian high (trend reversal)
                if high[i] > stop_level or close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    lowest_since_entry = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price > weekly HMA for long, price < weekly HMA for short
        price_above_weekly_hma = close[i] > hma_21_aligned[i]
        price_below_weekly_hma = close[i] < hma_21_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Long: Donchian breakout above upper band with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above Donchian high
            price_above_weekly_hma and       # Weekly trend alignment
            volume_spike                     # Volume confirmation
        )
        
        # Short: Donchian breakdown below lower band with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below Donchian low
            price_below_weekly_hma and       # Weekly trend alignment
            volume_spike                     # Volume confirmation
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = close[i]
            lowest_since_entry = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = close[i]
            lowest_since_entry = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</think>