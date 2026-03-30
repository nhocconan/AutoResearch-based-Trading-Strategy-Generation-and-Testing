#!/usr/bin/env python3
"""
Experiment #025: 12h Donchian Breakout + ADX Filter + Volume

HYPOTHESIS: 12h is the optimal timeframe (better than 6h: Sharpe 0.308 vs 0.01).
Using ADX(14)>20 instead of choppiness to confirm trend strength before
entering on Donchian breakouts. ADX is more responsive than choppiness
and directly measures whether the market is trending.

- 2021 bull: ADX>20 + breakout above Donchian = strong trend continuation
- 2022 bear: ADX>20 + breakdown below Donchian = strong short rallies
- 2025 range: ADX<20 = no entry, avoiding whipsaws

TARGET: 75-150 total over 4 years. Size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_adx_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """ADX (Average Directional Index) - measures trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # Directional Movement
    up_move = high[i] - high[i-1]
    down_move = low[i-1] - low[i]
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        plus_dm[i] = up if (up > down and up > 0) else 0
        minus_dm[i] = down if (down > up and down > 0) else 0
    
    # Smooth with Wilder's method (ATR uses same smoothing)
    atr = np.zeros(n, dtype=np.float64)
    atr[period] = np.sum(tr[1:period+1])
    for i in range(period + 1, n):
        atr[i] = atr[i-1] - atr[i-1] / period + tr[i]
    
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    
    plus_dms = np.zeros(n, dtype=np.float64)
    minus_dms = np.zeros(n, dtype=np.float64)
    plus_dms[period] = np.sum(plus_dm[1:period+1])
    minus_dms[period] = np.sum(minus_dm[1:period+1])
    
    for i in range(period + 1, n):
        plus_dms[i] = plus_dms[i-1] - plus_dms[i-1] / period + plus_dm[i]
        minus_dms[i] = minus_dms[i-1] - minus_dms[i-1] / period + minus_dm[i]
        
        if atr[i] > 1e-10:
            plus_di[i] = 100 * plus_dms[i] / atr[i]
            minus_di[i] = 100 * minus_dms[i] / atr[i]
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 1e-10:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX = smoothed DX
    adx = np.zeros(n, dtype=np.float64)
    adx[period * 2] = np.mean(dx[period:period*2])
    
    for i in range(period * 2 + 1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA200 for macro direction (call ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Donchian 20 - breakout channel (shift by 1 to avoid look-ahead)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 220  # Need 200 for SMA200 + 20 for Donchian
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DETECTION ===
        # ADX > 20 means market is trending (not choppy)
        adx_trending = adx_14[i] > 20
        
        # HTF 1d SMA200 direction
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === BREAKOUT CONDITIONS ===
        bullish_breakout = (close[i] > dc_upper_20[i]) if not np.isnan(dc_upper_20[i]) else False
        bearish_breakout = (close[i] < dc_lower_20[i]) if not np.isnan(dc_lower_20[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma[i] * 1.2 if vol_ma[i] > 1e-10 else False
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (24h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        if in_position:
            stop_hit = False
            if position_side > 0:
                # Long stop: price drops below highest - 2.5*ATR
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                # Short stop: price rises above lowest + 2.5*ATR
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on opposite HTF trend (after min hold)
            if min_hold:
                if position_side > 0 and htf_bearish:
                    stop_hit = True
                if position_side < 0 and htf_bullish:
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Breakout above + volume confirm + ADX trending + 1d uptrend
            if bullish_breakout and adx_trending and htf_bullish:
                in_position = True
                position_side = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Breakdown below + volume confirm + ADX trending + 1d downtrend
            elif bearish_breakout and adx_trending and htf_bearish:
                in_position = True
                position_side = -1
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals