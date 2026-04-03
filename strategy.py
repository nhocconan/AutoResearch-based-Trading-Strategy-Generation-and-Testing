#!/usr/bin/env python3
"""
Experiment #352: 12h Donchian Breakout + HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe, confirmed by 1d HMA trend alignment and 1w volume spike, 
captures strong momentum moves while filtering false breakouts. The Donchian channel provides objective 
breakout levels, HMA(21) on 1d ensures trend alignment to avoid counter-trend trades, and 1w volume 
confirmation (>2.0x average) ensures institutional participation. Targets 12-37 trades/year on 12h 
timeframe (50-150 total over 4 years) to minimize fee drag. ATR-based stoploss (2.5x) manages risk.
Works in both bull (breakouts continuation) and bear (breakdown continuation) markets.
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
        wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).mean().values
        wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume spike (Call ONCE before loop) ===
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
    # Calculate Donchian(20) channels on 12h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
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
        # Long: Price breaks above Donchian upper band + 1d HMA uptrend + 1w volume spike
        long_condition = (
            close[i] > highest_20[i] and  # Breakout above upper band
            close[i] > hma_21_1d_aligned[i] and  # Price above 1d HMA (uptrend)
            vol_ratio_1w_aligned[i] > 2.0  # 1w volume spike > 2.0x average
        )
        
        # Short: Price breaks below Donchian lower band + 1d HMA downtrend + 1w volume spike
        short_condition = (
            close[i] < lowest_20[i] and  # Breakdown below lower band
            close[i] < hma_21_1d_aligned[i] and  # Price below 1d HMA (downtrend)
            vol_ratio_1w_aligned[i] > 2.0  # 1w volume spike > 2.0x average
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals