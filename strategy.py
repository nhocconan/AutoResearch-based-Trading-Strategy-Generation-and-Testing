#!/usr/bin/env python3
"""
Experiment #137: 4h Donchian Breakout + 1d/1w Regime Filter + Volume Spike

HYPOTHESIS: Donchian(20) breakouts on 4h capture medium-term trends. 
1d/1w regime filter (ADX>25 = trend, ADX<20 = range) avoids whipsaws. 
In trend regimes: trade breakouts in trend direction. 
In range regimes: fade breakouts at extremes (mean reversion). 
Volume spike confirmation ensures institutional participation. 
ATR-based stoploss manages risk. 
Uses 4h timeframe for optimal trade frequency (target: 75-200 total trades over 4 years).
Works in bull/bear by adapting to regime: trend following in strong trends, mean reversion in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_breakout_1d_1w_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (ADX) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === HTF: 1w data for stronger regime filter (ADX) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate ADX function
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    # Calculate ADX for 1d and 1w
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align HTF data to LTF
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === 4h Indicators ===
    # Donchian Channel (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # ATR for stoploss and volatility filter
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Volume spike detection (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (volume_ma + 1e-10)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ratio[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter (Combined 1d and 1w ADX) ---
        adx_1d_val = adx_1d_aligned[i]
        adx_1w_val = adx_1w_aligned[i]
        
        # Strong trend: both timeframes show trending
        is_strong_trend = (adx_1d_val > 25) and (adx_1w_val > 25)
        # Weak trend/ranging: at least one timeframe shows ranging
        is_weak_or_range = (adx_1d_val < 20) or (adx_1w_val < 20)
        
        # --- Donchian Breakout Signals ---
        breakout_up = close[i] > donch_upper[i-1]  # Break above upper channel
        breakout_down = close[i] < donch_lower[i-1]  # Break below lower channel
        
        # --- Volume Confirmation ---
        volume_spike = volume_ratio[i] > 2.0  # Volume at least 2x average
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Stoploss: 2 * ATR against position
            if position_side > 0:  # Long position
                if close[i] < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if close[i] > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Exit conditions based on regime
            if is_strong_trend:
                # In strong trend: exit on opposite breakout or volume drying up
                if position_side > 0:  # Long
                    if breakout_down or volume_ratio[i] < 0.5:  # Opposite breakout or low volume
                        in_position = False
                        position_side = 0
                else:  # Short
                    if breakout_up or volume_ratio[i] < 0.5:  # Opposite breakout or low volume
                        in_position = False
                        position_side = 0
            else:  # weak trend or range
                # In range: exit when price returns to channel midpoint or opposite breakout
                midpoint = (donch_upper[i] + donch_lower[i]) / 2.0
                if position_side > 0:  # Long
                    if close[i] < midpoint or breakout_down:
                        in_position = False
                        position_side = 0
                else:  # Short
                    if close[i] > midpoint or breakout_up:
                        in_position = False
                        position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_strong_trend:
            # Strong trend regime: trade breakouts in trend direction
            if breakout_up and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            elif breakout_down and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        else:  # weak trend or range
            # Weak trend/range regime: fade breakouts (mean reversion)
            if breakout_up and volume_spike:
                # Break above upper channel -> expect reversion to mean (short)
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            elif breakout_down and volume_spike:
                # Break below lower channel -> expect reversion to mean (long)
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
    
    return signals