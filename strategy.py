#!/usr/bin/env python3
"""
Experiment #342: 12h Donchian(20) breakout + 1d HMA trend + 1w volume confirmation

HYPOTHESIS: Donchian channel breakouts on 12h timeframe capture significant price moves, 
filtered by 1d Hull Moving Average trend direction and 1w volume spike confirmation. 
This combination provides high-probability entries in both bull and bear markets by 
trading with the higher timeframe trend and institutional volume participation. 
Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize 
fee drag while capturing strong trending moves. Uses discrete position sizing (0.25) 
and ATR-based stoploss for risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === 12h Indicators ===
    # Calculate Donchian channels (20-period) on 12h
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_middle = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:  # Need 20 periods for lookback
            period_high = high[i-19:i+1].max()
            period_low = low[i-19:i+1].min()
            donchian_upper[i] = period_high
            donchian_lower[i] = period_low
            donchian_middle[i] = (period_high + period_low) / 2
        else:
            donchian_upper[i] = np.nan
            donchian_lower[i] = np.nan
            donchian_middle[i] = np.nan
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction (rising for long, falling for short) ---
        if i >= warmup + 1:
            hma_rising = hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1]
            hma_falling = hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.8
        
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
                # Take profit at Donchian middle (mean reversion)
                if close[i] <= donchian_middle[i]:
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
                # Take profit at Donchian middle (mean reversion)
                if close[i] >= donchian_middle[i]:
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
            close[i] > donchian_upper[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian lower with HMA falling and volume spike
        short_condition = (
            close[i] < donchian_lower[i] and 
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