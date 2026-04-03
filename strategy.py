#!/usr/bin/env python3
"""
Experiment #062: 12h Donchian Breakout + HMA Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Donchian(20) breakouts on 12h timeframe capture strong trends, confirmed by HMA(21) direction and volume spikes (>1.5x average). 
ATR(14) stoploss limits drawdown. Works in both bull/bear markets by trading breakouts in direction of higher timeframe trend (1d/1w HMA).
Target: 50-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_062_12h_donchian_hma_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Higher Timeframe Trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: HMA(21) for trend filter ===
    def calculate_hma(arr, period):
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        n = len(arr)
        hma = np.full(n, np.nan)
        
        if n < period:
            return hma
            
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        if len(wma_half) == 0 or len(wma_full) == 0:
            return hma
            
        # Align arrays: wma_half starts at index half_period-1, wma_full at period-1
        diff = 2 * wma_half - wma_full
        # diff starts at index max(half_period-1, period-1) = period-1
        wma_diff = wma(diff, sqrt_period)
        
        if len(wma_diff) == 0:
            return hma
            
        # Place wma_diff result at correct position
        start_idx = period - 1 + sqrt_period - 1
        end_idx = start_idx + len(wma_diff)
        if end_idx <= n:
            hma[start_idx:end_idx] = wma_diff
            
        return hma
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === HTF: 1w data for Higher Timeframe Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # === 12h Indicators: Donchian Channels (20) ===
    def calculate_donchian(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, 20)
    
    # === 12h Indicators: HMA(21) for entry confirmation ===
    hma_12h = calculate_hma(close, 21)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 12h Indicators: ATR(14) for stoploss ===
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # Warmup for indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]) or 
            np.isnan(donchian_20_upper[i]) or np.isnan(donchian_20_lower[i]) or
            np.isnan(hma_12h[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = vol_ratio[i] > 1.5  # Volume spike threshold
        
        # Higher timeframe trend filter: both 1d and 1w HMA must agree
        htf_bullish = hma_1d_aligned[i] > hma_1d_aligned[i-1] and hma_1w_aligned[i] > hma_1w_aligned[i-1]
        htf_bearish = hma_1d_aligned[i] < hma_1d_aligned[i-1] and hma_1w_aligned[i] < hma_1w_aligned[i-1]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # Stoploss: 2.5 * ATR against position
            if position_side > 0 and price < entry_price - 2.5 * atr[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price > entry_price + 2.5 * atr[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            # Take profit: reduce to half at 2R profit
            if position_side > 0 and price >= entry_price + 5.0 * atr[i]:  # 2R profit
                signals[i] = SIZE * 0.5  # Half position
                continue
            elif position_side < 0 and price <= entry_price - 5.0 * atr[i]:  # 2R profit
                signals[i] = -SIZE * 0.5  # Half position
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: Donchian breakout above upper band + HMA up + volume spike + HTF bullish
        if (price > donchian_20_upper[i-1] and 
            hma_12h[i] > hma_12h[i-1] and 
            vol_spike and 
            htf_bullish):
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short: Donchian breakout below lower band + HMA down + volume spike + HTF bearish
        elif (price < donchian_20_lower[i-1] and 
              hma_12h[i] < hma_12h[i-1] and 
              vol_spike and 
              htf_bearish):
            in_position = True
            position_side = -1
            entry_price = close[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals