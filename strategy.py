#!/usr/bin/env python3
"""
Experiment #324: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Daily Donchian channel breakouts (20-period) aligned with weekly HMA(21) trend direction, 
confirmed by daily volume spikes (>2x 20-day average), produce high-probability trades in both 
bull and bear markets. The weekly timeframe provides robust trend filtering to avoid counter-trend 
whipsaws, while daily breaks capture momentum. ATR-based stoploss (2.5x) manages risk. Targets 
15-25 trades/year on 1d timeframe (60-100 total over 4 years) to minimize fee drag while 
participating in strong trending moves. Works in bear markets via short signals on breakdowns 
below Donchian lower band when weekly trend is down.
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
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        # Use rolling window with min_periods
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
    
    # Volume ratio (current vs 20-day average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_series = pd.Series(volume)
        vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = vol_series.values[20:] / vol_ma_20[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Weekly HMA direction ---
        # Need previous HMA value to determine slope
        if i == warmup:
            prev_hma = hma_21_aligned[i-1]
        else:
            prev_hma = hma_21_aligned[i-1]
        curr_hma = hma_21_aligned[i]
        hma_rising = curr_hma > prev_hma
        hma_falling = curr_hma < prev_hma
        
        # --- Volume Confirmation: Require volume spike (> 2x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
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
                # Exit if price crosses below weekly HMA (trend change)
                if close[i] < hma_21_aligned[i]:
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
                # Exit if price crosses above weekly HMA (trend change)
                if close[i] > hma_21_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume + weekly uptrend
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            hma_rising
        )
        
        # Short: Price breaks below Donchian lower band with volume + weekly downtrend
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            hma_falling
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