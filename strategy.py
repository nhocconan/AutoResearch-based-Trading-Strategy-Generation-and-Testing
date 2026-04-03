#!/usr/bin/env python3
"""
Experiment #089: 4h Donchian(20) Breakout + HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d HMA trend direction and 
12h volume spike confirmation, creates a robust strategy that works in both bull and bear markets. 
The Donchian structure captures breakouts from key support/resistance levels, HMA provides smooth 
trend direction without lag, and volume confirms institutional participation. Targets 25-50 
trades/year on 4h timeframe (100-200 total over 4 years) to minimize fee drag while capturing 
high-probability breakouts with proper risk management via ATR-based stoploss.
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
    
    # === HTF: 12h data for volume spike confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # Calculate HMA(21) on 1d close: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        # Use rolling window with min_periods=20 to avoid look-ahead
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
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
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(hma_21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in direction of 1d HMA trend ---
        price_above_1d_hma = close[i] > hma_21_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_21_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using only past data
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
                # Exit on Donchian lower band touch (mean reversion within trend)
                if close[i] <= donchian_low[i]:
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
                # Exit on Donchian upper band touch (mean reversion within trend)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian upper band with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper band
            volume_spike and                 # Volume confirmation
            price_above_1d_hma               # Trend filter: bullish
        )
        
        # Short: Break below Donchian lower band with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below lower band
            volume_spike and                 # Volume confirmation
            price_below_1d_hma               # Trend filter: bearish
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