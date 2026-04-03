#!/usr/bin/env python3
"""
Experiment #424: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 1d timeframe capture significant momentum moves, 
filtered by 1w HMA(21) trend direction and confirmed by 1d volume spikes (>2.0x average). 
This combination provides high-probability entries in both bull and bear markets by trading 
with the higher timeframe trend. ATR-based stoploss (2.5x) manages risk. Targets 75-250 
total trades over 4 years (19-62/year) to minimize fee drag while capturing strong moves.
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
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
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
    # Donchian Channel (20) - using previous 20 periods for breakout
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        # Use rolling window on previous 20 bars (excluding current)
        donchian_high[20:] = high_series.rolling(window=20, min_periods=20).max().shift(1).values[20:]
        donchian_low[20:] = low_series.rolling(window=20, min_periods=20).min().shift(1).values[20:]
        # For first 20 bars, use expanding window
        for i in range(20):
            donchian_high[i] = high_series.iloc[:i+1].max()
            donchian_low[i] = low_series.iloc[:i+1].min()
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, 1.0)
    if n >= 20:
        volume_series = pd.Series(volume)
        vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        # Need previous HMA value to determine slope
        if i > 0 and not np.isnan(hma_21_aligned[i-1]):
            hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
            hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using only past data
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian High + HMA rising + volume spike
        long_condition = (
            close[i] > donchian_high[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian Low + HMA falling + volume spike
        short_condition = (
            close[i] < donchian_low[i] and 
            hma_falling and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</file>