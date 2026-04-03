#!/usr/bin/env python3
"""
Experiment #077: 4h Donchian Breakout + HMA Trend + Volume Filter (1d HTF)
HYPOTHESIS: 4h Donchian(20) breakouts in the direction of 1d HMA(21) trend with volume confirmation (>1.5x average) capture strong momentum moves in both bull and bear markets. The 1d HMA filter ensures we only trade with the higher timeframe trend, reducing whipsaws. Volume confirmation adds conviction to breakouts. Target: 100-180 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_077_4h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA(21) for trend direction ===
    def calculate_hma(arr, period):
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half = period // 2
        sqrt = int(np.sqrt(period))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        wma_2half_minus_full = 2 * wma_half - wma_full
        hma = wma(wma_2half_minus_full, sqrt)
        
        # Pad with NaN to match original length
        result = np.full_like(arr, np.nan)
        result[half:half + len(hma)] = hma
        return result
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 4h Indicators: Donchian Channels (20) ===
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Warmup for Donchian and volume stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # --- Trend Direction from 1d HMA ---
        is_uptrend = hma_1d_aligned[i] > hma_1d_aligned[i-1]  # HMA rising
        is_downtrend = hma_1d_aligned[i] < hma_1d_aligned[i-1]  # HMA falling
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2.5 * ATR(14) approximation using 20-period range
            atr_approx = (donchian_upper[i] - donchian_lower[i]) / 2.0
            stoploss_long = entry_price - 2.5 * atr_approx
            stoploss_short = entry_price + 2.5 * atr_approx
            
            # Exit conditions
            if position_side > 0:  # Long
                if price < stoploss_long or price < donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price > stoploss_short or price > donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: Donchian breakout above upper band in uptrend with volume
        if price > donchian_upper[i-1] and is_uptrend and vol_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian breakout below lower band in downtrend with volume
        elif price < donchian_lower[i-1] and is_downtrend and vol_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals