#!/usr/bin/env python3
"""
Experiment #025: 6h Regime-Adaptive Dual Strategy

HYPOTHESIS: A single strategy that adapts to market regime should work in BOTH
bull (2021) and bear (2022) markets:
- Trending regime (ADX > 25): Use Donchian breakout + volume = captures big moves
- Choppy regime (ADX < 20): Use Bollinger Band mean reversion = profits from ranges

KEY INSIGHT: Previous 6h attempts failed because:
1. Strict HTF filter (SMA) conflicted with 6h timing
2. Single strategy doesn't adapt to regime changes

This approach: Auto-detect regime, apply appropriate strategy.
Should generate 75-150 trades (18-37/year) with SIZE=0.30.

TARGET: 75-150 total over 4 years. Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_regime_adaptive_dual_v1"
timeframe = "6h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                   abs(high[i] - close[i-1]), 
                   abs(low[i] - close[i-1]))
    
    # Directional Movement
    up_move = np.zeros(n)
    down_move = np.zeros(n)
    
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        
        if up > down and up > 0:
            up_move[i] = up
        if down > up and down > 0:
            down_move[i] = down
    
    # Smoothed values using Wilder's method
    atr_smooth = np.zeros(n)
    atr_smooth[period-1] = np.sum(tr[1:period])
    for i in range(period, n):
        atr_smooth[i] = atr_smooth[i-1] - atr_smooth[i-1]/period + tr[i]
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    plus_dm_smooth[period-1] = np.sum(up_move[1:period])
    minus_dm_smooth[period-1] = np.sum(down_move[1:period])
    
    for i in range(period, n):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - plus_dm_smooth[i-1]/period + up_move[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - minus_dm_smooth[i-1]/period + down_move[i]
        
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
    
    # DX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX as smoothed DX
    adx = np.full(n, np.nan)
    adx[2*period-1] = np.mean(dx[period:2*period])
    
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
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
    
    # === HTF: 1d SMA for macro direction (call ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Donchian 20 for breakout
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Bollinger Bands for mean reversion
    bb_sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    
    # Volume confirmation (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 60  # Need enough for ADX calculation
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        adx_value = adx_14[i]
        trending = adx_value > 25   # Strong trend
        choppy = adx_value < 20    # Range-bound
        
        # === HTF TREND (weaker filter - just for direction bias) ===
        htf_bullish = close[i] > sma_1d_aligned[i] if not np.isnan(sma_1d_aligned[i]) else True
        htf_bearish = close[i] < sma_1d_aligned[i] if not np.isnan(sma_1d_aligned[i]) else False
        
        # === BREAKOUT CONDITIONS ===
        bullish_breakout = close[i] > dc_upper_20[i] if not np.isnan(dc_upper_20[i]) else False
        bearish_breakout = close[i] < dc_lower_20[i] if not np.isnan(dc_lower_20[i]) else False
        
        # === BB MEAN REVERSION CONDITIONS ===
        at_lower_band = close[i] < bb_lower[i] if not np.isnan(bb_lower[i]) else False
        at_upper_band = close[i] > bb_upper[i] if not np.isnan(bb_upper[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma[i] * 1.2 if vol_ma[i] > 1e-10 else False
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.0x ATR) ===
        if in_position:
            stop_hit = False
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.0 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.0 * atr_14[i])
            
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
            # TRENDING REGIME: Donchian breakout + volume + HTF direction
            if trending:
                # LONG: Breakout + volume + HTF bullish
                if bullish_breakout and vol_ok and htf_bullish:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    highest_since_entry = high[i]
                    signals[i] = SIZE
                
                # SHORT: Breakdown + volume + HTF bearish
                elif bearish_breakout and vol_ok and htf_bearish:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            
            # CHOPPY REGIME: Bollinger Band mean reversion
            elif choppy:
                # LONG: Price at lower BB + HTF bullish
                if at_lower_band and htf_bullish:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    highest_since_entry = high[i]
                    signals[i] = SIZE
                
                # SHORT: Price at upper BB + HTF bearish
                elif at_upper_band and htf_bearish:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            
            # NEUTRAL ZONE: No trade
            else:
                signals[i] = 0.0
    
    return signals