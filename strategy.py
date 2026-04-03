#!/usr/bin/env python3
"""
Experiment #277: 4h Donchian(20) breakout + 1d HMA trend + 1w volume confirmation

HYPOTHESIS: Trading Donchian channel breakouts on 4h with alignment to 1d HMA trend and 1w volume spike captures strong momentum moves while avoiding false breakouts. The 4h timeframe targets 25-50 trades/year (100-200 total over 4 years) to minimize fee drag. Volume confirmation on weekly ensures institutional participation. Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by trading both directions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_1dhma_1wvol_v1"
timeframe = "4h"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        # Handle array lengths
        wma_half = np.concatenate([np.full(len(close_1d) - len(wma_half), np.nan), wma_half])
        wma_full = np.concatenate([np.full(len(close_1d) - len(wma_full), np.nan), wma_full])
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = wma(raw_hma, sqrt_len)
        hma_21_1d = np.concatenate([np.full(len(close_1d) - len(hma_21_1d), np.nan), hma_21_1d])
        
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Volume SMA(10) on 1w volume
    if len(df_1w) >= 10:
        volume_1w = df_1w['volume'].values
        vol_sma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
        vol_sma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma_10_1w)
    else:
        vol_sma_10_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_sma_10_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Current 4h volume > 1.5x weekly average volume per 4h bar ---
        # Approximate: weekly volume / (7*24/4) = weekly volume / 42 (number of 4h bars in week)
        weekly_avg_per_4h = vol_sma_10_1w_aligned[i] / 42.0
        volume_spike = volume[i] > (1.5 * weekly_avg_per_4h)
        
        # --- Donchian Breakout Conditions ---
        breakout_long = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakdown_short = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # --- 1d HMA Trend Filter ---
        price_above_hma = close[i] > hma_21_1d_aligned[i]
        price_below_hma = close[i] < hma_21_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3x ATR profit
                if close[i] > close[entry_bar] + 3.0 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3x ATR profit
                if close[i] < close[entry_bar] - 3.0 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout + above 1d HMA + volume spike
        if breakout_long and price_above_hma and volume_spike:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakdown + below 1d HMA + volume spike
        elif breakdown_short and price_below_hma and volume_spike:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals