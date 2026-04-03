#!/usr/bin/env python3
"""
Experiment #413: 4h Donchian(20) Breakout + 12h Volume Spike + 12h HMA Trend Filter

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 12h volume spike (>2x average) 
and 12h HMA(21) trend alignment, capture high-probability momentum moves in both bull and bear markets. 
The Donchian structure provides objective breakout levels, volume confirms institutional participation, 
and the HMA trend filter ensures we only trade with the higher timeframe momentum. Targets 25-50 
trades/year on 4h timeframe (100-200 total over 4 years) to minimize fee drag while capturing 
significant breakouts. Uses ATR(14) stoploss and discrete position sizing (0.30) to control risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_vol_hma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike and HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        if len(wma_half) >= half_len and len(wma_full) >= 21:
            wma_2x_sub = 2 * wma_half[-len(wma_full):] - wma_full
            hma_values = wma(wma_2x_sub, sqrt_len)
            # Pad to match original length
            hma_12h = np.full(len(close_12h), np.nan)
            hma_12h[-len(hma_values):] = hma_values
            hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
        else:
            hma_12h_aligned = np.full(n, np.nan)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Calculate Donchian channels (20-period) on 4h
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    if n >= donchian_period:
        # Use pandas rolling for efficiency
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        upper_channel = high_series.rolling(window=donchian_period, min_periods=donchian_period).max().values
        lower_channel = low_series.rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    atr = np.full(n, np.nan)
    if n >= atr_period:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for j in range(1, n):
            tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        atr_series = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean()
        atr = atr_series.values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position sizing (30% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_period, atr_period, 20)  # Ensure enough data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio_12h_aligned[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(close_12h := df_12h['close'].values[-1] if len(df_12h) > 0 else np.nan)):
            signals[i] = 0.0
            continue
        
        # Get current 12h close for trend comparison (simplified: use aligned HMA vs price)
        # We'll use the 12h HMA slope approximation: current HMA vs previous HMA
        if i >= warmup + 1:
            hma_now = hma_12h_aligned[i]
            hma_prev = hma_12h_aligned[i-1]
            hma_rising = hma_now > hma_prev
            hma_falling = hma_now < hma_prev
        else:
            hma_rising = hma_falling = False
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above upper Donchian channel + volume spike + HMA rising
        long_condition = (
            close[i] > upper_channel[i] and  # Breakout above channel
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike (>2x average)
            hma_rising  # 12h HMA trending up
        )
        
        # Short: Price breaks below lower Donchian channel + volume spike + HMA falling
        short_condition = (
            close[i] < lower_channel[i] and  # Breakdown below channel
            vol_ratio_12h_aligned[i] > 2.0 and  # Volume spike (>2x average)
            hma_falling  # 12h HMA trending down
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