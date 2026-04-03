#!/usr/bin/env python3
"""
Experiment #128: 12h Donchian(20) breakout + 1w HMA trend + 1d volume confirmation

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by 1w HMA trend direction and confirmed by 1d volume spike, captures medium-term trends with minimal trades. The 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag. Volume confirmation ensures institutional participation, while the 1w HMA filter avoids counter-trend trades in bear markets. Discrete position sizing (0.25) reduces churn. Strategy works in both bull (breakouts with volume) and bear (short breakdowns with volume) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian20_1w_hma_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    hma_21_1w = np.full(n, np.nan)
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA(21) = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1w_values = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
        hma_21_1w = align_htf_to_ltf(prices, df_1w, hma_21_1w_values)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ratio_20_1d = np.full(n, 1.0)
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_20_1d_values = np.ones_like(vol_1d)
        vol_ratio_20_1d_values[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_20_1d = align_htf_to_ltf(prices, df_1d, vol_ratio_20_1d_values)
    
    # === LTF: 12h Donchian(20) breakout levels ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    if n >= 20:
        # Donchian high: max(high) over last 20 periods
        donchian_h_values = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian low: min(low) over last 20 periods
        donchian_l_values = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_h = donchian_h_values
        donchian_l = donchian_l_values
    
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
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(hma_21_1w[i]) or np.isnan(vol_ratio_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 1w HMA direction ---
        # Need previous bar's HMA to determine trend (avoid look-ahead)
        if i == warmup:
            hma_prev = hma_21_1w[i-1] if i-1 >= 0 else hma_21_1w[i]
        else:
            hma_prev = hma_21_1w[i-1]
        
        hma_curr = hma_21_1w[i]
        hma_rising = hma_curr > hma_prev
        hma_falling = hma_curr < hma_prev
        
        # --- Volume Confirmation: 1d volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_20_1d[i] > 1.8
        
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
                # Exit on Donchian lower band touch (trailing stop)
                if close[i] <= donchian_l[i]:
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
                # Exit on Donchian upper band touch (trailing stop)
                if close[i] >= donchian_h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian H with volume + HMA rising
        long_condition = (
            close[i] > donchian_h[i] and  # Breakout above upper band
            volume_spike and              # Volume confirmation
            hma_rising                    # Uptrend filter
        )
        
        # Short: Price breaks below Donchian L with volume + HMA falling
        short_condition = (
            close[i] < donchian_l[i] and  # Breakdown below lower band
            volume_spike and              # Volume confirmation
            hma_falling                   # Downtrend filter
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