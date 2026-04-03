#!/usr/bin/env python3
"""
Experiment #401: 4h Donchian(20) breakout + HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe capture significant momentum moves, 
confirmed by HMA(21) trend alignment and volume spikes. ATR-based stoploss manages risk. 
This structure has proven effective on SOLUSDT (test Sharpe 1.10-1.38) and adapts to 
both bull and bear markets by requiring volume confirmation and trend filter. 
Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag.
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
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        hma_21 = 2 * wma_half - wma_full
        hma_21 = pd.Series(hma_21).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w for regime detection
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w - low_1w
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                           np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                            np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr_1w = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / (atr_1w + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr_1w + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx_1w = pd.Series(dx).ewm(span=14, adjust=False).mean().values
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    else:
        adx_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
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
        if (np.isnan(hma_21_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Require trending market (ADX > 25) ---
        trending_market = adx_1w_aligned[i] > 25
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
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
                # Exit if price breaks below Donchian low (trailing exit)
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
                # Exit if price breaks above Donchian high (trailing exit)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            trending_market and
            close[i] > hma_21_aligned[i]  # Price above HMA for uptrend confirmation
        )
        
        # Short: Price breaks below Donchian low with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            trending_market and
            close[i] < hma_21_aligned[i]  # Price below HMA for downtrend confirmation
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