#!/usr/bin/env python3
"""
Experiment #364: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Daily Donchian(20) breakouts capture medium-term trends, while 1-week HMA(21) 
filters for alignment with higher timeframe direction. Volume confirmation (>1.5x 20-day avg) 
ensures institutional participation. This combination works in both bull and bear markets by 
trading breakouts in the direction of the weekly trend. Targets 20-40 trades/year on 1d 
timeframe (80-160 total over 4 years) to minimize fee drag while capturing high-probability 
breakouts with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_vol_1w_trend_v2"
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
        # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights/weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            wma_2x_sub = 2 * wma_half - wma_full[-len(wma_half):]
            hma_21 = wma(wma_2x_sub, sqrt_len)
            # Pad to match original length
            hma_21_padded = np.full(len(close_1w), np.nan)
            hma_21_padded[half_len:-sqrt_len+1 if sqrt_len>1 else None] = hma_21
            hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
        else:
            hma_21_aligned = np.full(n, np.nan)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[20:] = high_series.rolling(window=20, min_periods=20).max().values[20:]
        donchian_low[20:] = low_series.rolling(window=20, min_periods=20).min().values[20:]
    
    # Volume ratio (current vs 20-day average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_series = pd.Series(volume)
        vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
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
        
        # --- Trend Filter: HMA direction (rising for long, falling for short) ---
        if i >= warmup + 1:
            hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
            hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
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
                # Exit on Donchian low break (trailing stop)
                if close[i] < donchian_low[i]:
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
                # Exit on Donchian high break (trailing stop)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian high breakout + rising HMA + volume spike
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper band
            hma_rising and                   # Weekly trend up
            volume_spike                     # Volume confirmation
        )
        
        # Short: Donchian low breakdown + falling HMA + volume spike
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below lower band
            hma_falling and                  # Weekly trend down
            volume_spike                     # Volume confirmation
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