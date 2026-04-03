#!/usr/bin/env python3
"""
Experiment #378: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on daily timeframe capture strong momentum moves, 
filtered by weekly HMA trend direction and volume confirmation to avoid false breakouts. 
This structure works in both bull and bear markets by trading with the higher timeframe 
trend while using volume to confirm institutional participation. Targets 7-25 trades/year 
on 1d timeframe (30-100 total over 4 years) to minimize fee drag while capturing 
high-probability trend continuation moves.
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
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).rolling(window=half_len, min_periods=half_len).mean().values
        wma_full = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Calculate Donchian channels (20-period) on 1d
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    if n >= donchian_period:
        # Use rolling max/min for Donchian channels
        upper_channel[donchian_period-1:] = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values[donchian_period-1:]
        lower_channel[donchian_period-1:] = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values[donchian_period-1:]
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    vol_ratio_1d = np.full(n, np.nan)
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d[20:] = volume[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
    
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
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d[i])):
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
        volume_spike = vol_ratio_1d[i] > 1.5
        
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
                # Exit if price crosses below Donchian lower (trend exhaustion)
                if close[i] < lower_channel[i]:
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
                # Exit if price crosses above Donchian upper (trend exhaustion)
                if close[i] > upper_channel[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper with HMA rising and volume spike
        long_condition = (
            close[i] > upper_channel[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian lower with HMA falling and volume spike
        short_condition = (
            close[i] < lower_channel[i] and 
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