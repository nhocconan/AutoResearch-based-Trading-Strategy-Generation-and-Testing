#!/usr/bin/env python3
"""
Experiment #377: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe, filtered by HMA(21) trend direction and 
1d volume spike confirmation, captures strong momentum moves in both bull and bear markets. 
The Donchian channel provides objective breakout levels, HMA ensures trend alignment, 
volume confirms institutional participation, and ATR-based stoploss manages risk. 
Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag while capturing 
high-probability trend continuation moves. Works in bull markets via breakouts and in 
bear markets via breakdowns with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h Indicators ===
    # Calculate HMA(21) on 4h close
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).mean().values
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
        return hma
    
    hma_21 = calculate_hma(close, 21)
    
    # Calculate Donchian(20) channels on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        donchian_high[20:] = pd.Series(high).rolling(window=20, min_periods=20).max().values[20:]
        donchian_low[20:] = pd.Series(low).rolling(window=20, min_periods=20).min().values[20:]
        # For warmup period, use expanding window
        for i in range(20):
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    max_favorable_price = 0.0  # For trailing stop
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss with trailing) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update max favorable price
                max_favorable_price = max(max_favorable_price, high[i])
                # Initial stoploss
                stop_level = entry_price - 2.0 * atr_14
                # Trailing stop: trail by 1.5 * ATR from max favorable price
                trail_stop = max_favorable_price - 1.5 * atr_14
                # Use the higher of initial stop and trail stop
                effective_stop = max(stop_level, trail_stop)
                if low[i] < effective_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian low break (mean reversion signal)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update max favorable price (lowest price reached)
                max_favorable_price = min(max_favorable_price, low[i])
                # Initial stoploss
                stop_level = entry_price + 2.0 * atr_14
                # Trailing stop: trail by 1.5 * ATR from max favorable price
                trail_stop = max_favorable_price + 1.5 * atr_14
                # Use the lower of initial stop and trail stop
                effective_stop = min(stop_level, trail_stop)
                if high[i] > effective_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian high break (mean reversion signal)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # Long: Donchian breakout above upper band with HMA uptrend and volume
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above Donchian high
            hma_21[i] > hma_21[i-1] and    # HMA rising (uptrend)
            volume_spike                   # Volume confirmation
        )
        
        # Short: Donchian breakdown below lower band with HMA downtrend and volume
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below Donchian low
            hma_21[i] < hma_21[i-1] and    # HMA falling (downtrend)
            volume_spike                   # Volume confirmation
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            max_favorable_price = high[i]  # Initialize trailing stop reference
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            max_favorable_price = low[i]   # Initialize trailing stop reference
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals