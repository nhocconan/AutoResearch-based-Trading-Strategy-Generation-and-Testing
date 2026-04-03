#!/usr/bin/env python3
"""
Experiment #1924: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture institutional accumulation/distribution phases. 
Weekly HMA(21) filter ensures we only trade with the dominant higher timeframe trend, reducing whipsaw.
Volume confirmation (>1.5x 20-day average) ensures breakouts have genuine participation.
ATR-based stoploss (2.5x ATR(14)) manages risk. Discrete position sizing (0.25) minimizes fee churn.
Target: 30-80 trades over 4 years (7-20/year) for statistical validity and low fee drag.
Works in both bull and bear markets by following the weekly trend while capturing daily momentum bursts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1924_1d_donchian20_1w_hma_vol_v1"
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
    close_1w = df_1w['close'].values
    
    # Calculate Weekly HMA(21): Hull Moving Average
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        hma_raw = 2 * wma_half - wma_full
        hma = pd.Series(hma_raw).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    
    hma_21_1w = calculate_hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Indicators: Donchian(20), ATR(14), Volume MA(20) ===
    # Donchian Channel: 20-period high/low
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0  # 1 for long, -1 for short
    entry_price = 0.0
    entry_atr = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), HMA(21)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2.5 * ATR against position
            stoploss_hit = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.5 * entry_atr:
                    stoploss_hit = True
            else:  # Short position
                if price > entry_price + 2.5 * entry_atr:
                    stoploss_hit = True
            
            # Exit on Donchian opposite touch (mean reversion)
            donchian_exit = False
            if position_side > 0:  # Long exit at lower Donchian
                if price <= low_roll[i]:
                    donchian_exit = True
            else:  # Short exit at upper Donchian
                if price >= high_roll[i]:
                    donchian_exit = True
            
            if stoploss_hit or donchian_exit:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Weekly trend filter: price above/below HMA(21)
        weekly_uptrend = close_1w[-1] > hma_21_1w[-1] if len(close_1w) == len(hma_21_1w) else False
        # Use aligned weekly trend for current bar
        # Simpler: compare current weekly close to its HMA
        # We need to get the weekly close value for the aligned weekly bar
        # Since we can't easily get it inside loop without storing, we use price vs aligned HMA as proxy
        # Better approach: use the fact that aligned HMA represents the completed weekly trend
        weekly_uptrend = price > hma_21_1w_aligned[i]
        weekly_downtrend = price < hma_21_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND weekly uptrend
            if weekly_uptrend and price > high_roll[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND weekly downtrend
            elif weekly_downtrend and price < low_roll[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals