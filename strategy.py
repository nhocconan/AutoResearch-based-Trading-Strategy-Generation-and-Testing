#!/usr/bin/env python3
"""
Experiment #073: 4h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts in direction of 12h HMA(21) trend with volume confirmation (>1.5x average)
capture strong momentum moves. ATR(14) stoploss limits drawdown. Works in bull/bear by following 12h trend.
Target: 75-200 trades over 4 years on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_073_4h_donchian_hma_vol_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h Indicators: HMA(21) for trend direction ===
    def calculate_hma(arr, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        def wma(values, window):
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        
        wma_half = np.convolve(arr, np.arange(1, half_period + 1), mode='valid') / (half_period * (half_period + 1) / 2)
        wma_full = np.convolve(arr, np.arange(1, period + 1), mode='valid') / (period * (period + 1) / 2)
        
        # Align arrays
        wma_half = np.concatenate([np.full(period - half_period, np.nan), wma_half])
        raw_hma = 2 * wma_half - wma_full
        wma_raw = np.convolve(raw_hma, np.arange(1, sqrt_period + 1), mode='valid') / (sqrt_period * (sqrt_period + 1) / 2)
        hma = np.concatenate([np.full(len(arr) - len(wma_raw), np.nan), wma_raw])
        return hma
    
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    hma_12h = calculate_hma(c_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Warmup for Donchian, ATR, HMA stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume confirmation threshold
        
        # --- Update position extremes for trailing stop ---
        if in_position:
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # --- Stoploss Logic ---
        if in_position:
            if position_side > 0:  # Long stop
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    highest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
            else:  # Short stop
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    lowest_since_entry = 0.0
                    signals[i] = 0.0
                    continue
        
        # --- Entry Logic ---
        if not in_position:
            # Long: price breaks above Donchian upper + 12h HMA uptrend + volume
            if (price > donchian_upper[i-1] and 
                hma_12h_aligned[i] > hma_12h_aligned[i-1] and  # HMA rising
                vol_spike):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                signals[i] = SIZE
            # Short: price breaks below Donchian lower + 12h HMA downtrend + volume
            elif (price < donchian_lower[i-1] and 
                  hma_12h_aligned[i] < hma_12h_aligned[i-1] and  # HMA falling
                  vol_spike):
                in_position = True
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # Hold position
            signals[i] = position_side * SIZE
    
    return signals