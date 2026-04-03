#!/usr/bin/env python3
"""
Experiment #083: 4h Donchian(20) breakout + 12h HMA trend + 1d volume confirmation

HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture significant momentum moves. 
Combined with 12h Hull Moving Average (HMA) trend filter to ensure alignment with higher 
timeframe direction and 1d volume spike confirmation to validate institutional participation, 
this strategy aims to catch strong trending moves while avoiding false breakouts in choppy 
markets. Uses ATR-based stoploss for risk management. Targets 19-50 trades/year on 4h 
timeframe (75-200 total over 4 years) to minimize fee drag while capturing high-probability 
breakouts. Designed to work in both bull and bear markets by using trend filter and volume 
confirmation to avoid whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_hma_1d_vol_v1"
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
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = np.full_like(close_12h, np.nan)
        wma_full = np.full_like(close_12h, np.nan)
        if len(close_12h) >= half_len:
            wma_half[half_len-1:] = wma(close_12h, half_len)
        if len(close_12h) >= 21:
            wma_full[20:] = wma(close_12h, 21)
        
        raw_hma = 2 * wma_half - wma_full
        hma_12h = np.full_like(close_12h, np.nan)
        if len(raw_hma) >= sqrt_len:
            hma_12h[sqrt_len-1:] = wma(raw_hma[sqrt_len-1:], sqrt_len)
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
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
    # Donchian Channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_high[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
        donchian_low[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
    
    # ATR(14) for stoploss
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price > 12h HMA for long, price < 12h HMA for short
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # Long: Donchian breakout above upper band with trend and volume
        long_condition = (
            close[i] > donchian_high[i] and 
            price_above_12h_hma and 
            volume_spike
        )
        
        # Short: Donchian breakdown below lower band with trend and volume
        short_condition = (
            close[i] < donchian_low[i] and 
            price_below_12h_hma and 
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