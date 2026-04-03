#!/usr/bin/env python3
"""
Experiment #030: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Daily Donchian breakouts capture significant momentum moves. Combined with 
weekly HMA trend filter and daily volume confirmation (>1.5x average), this strategy 
avoids false breakouts. Uses discrete position sizing (0.25) and ATR-based stoploss (2.5x ATR).
Designed to work in both bull (breakouts continuation) and bear (breakdowns continuation) markets.
Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_hma_1w_volume_v1"
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
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean()
        wma_diff = 2 * wma_half - wma_full
        hma = pd.Series(wma_diff).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        hma_21_1w = calculate_hma(close_1w, 21)
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === LTF: Daily Donchian channels (20-period) ===
    donchian_h20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === LTF: Daily volume confirmation (1.5x 20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
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
        if (np.isnan(hma_21_1w_aligned[i]) or 
            np.isnan(donchian_h20[i]) or np.isnan(donchian_l20[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 1week HMA21 ---
        price_above_1w_hma = close[i] > hma_21_1w_aligned[i]
        price_below_1w_hma = close[i] < hma_21_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume > 1.5x average ---
        volume_confirm = vol_ratio[i] > 1.5
        
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
        # Long: Price breaks above Donchian H20 with volume confirmation and uptrend
        long_condition = (
            close[i] > donchian_h20[i] and 
            volume_confirm and 
            price_above_1w_hma
        )
        
        # Short: Price breaks below Donchian L20 with volume confirmation and downtrend
        short_condition = (
            close[i] < donchian_l20[i] and 
            volume_confirm and 
            price_below_1w_hma
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