#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian Breakout + Williams %R Momentum + Volume

HYPOTHESIS: Donchian(20) breakout defines structure, Williams %R(14) at 
extremes (<30 for longs, >70 for shorts) confirms momentum reversal, 
loose volume filter (1.1x avg) eliminates low-volume false breakouts.
4h timeframe targets 75-150 total trades over 4 years (18-37/year).

WHY: Williams %R at extremes catches reversals at support/resistance while
Donchian defines the trading range. This combination filters choppy breakouts
while allowing trades during both 2021 bull and 2022 bear markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_wr_momentum_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    wr = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high - lowest_low > 1e-10:
            wr[i] = -100.0 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return wr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA(50) for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    wr_14 = calculate_williams_r(high, low, close, period=14)
    
    # Donchian 20 - price channel structure
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (20-bar average, loose filter 1.1x)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(wr_14[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND FILTER ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === BREAKOUT CONDITIONS ===
        # Price breaks above 20-bar high = bullish breakout
        bullish_breakout = (close[i] > dc_upper_20[i]) if not np.isnan(dc_upper_20[i]) else False
        # Price breaks below 20-bar low = bearish breakout
        bearish_breakout = (close[i] < dc_lower_20[i]) if not np.isnan(dc_lower_20[i]) else False
        
        # === MOMENTUM CONFIRMATION (Williams %R) ===
        # Long: %R below 30 = oversold momentum (reversal bounce)
        # Short: %R above 70 = overbought momentum (reversal dump)
        wr_oversold = wr_14[i] < -70  # -70 to -100 range
        wr_overbought = wr_14[i] > -30  # -30 to 0 range
        
        # === VOLUME CONFIRMATION (loose: 1.1x average) ===
        vol_ok = volume[i] > vol_ma[i] * 1.1 if vol_ma[i] > 1e-10 else False
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (8h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR from highest/lowest) ===
        if in_position:
            stop_hit = False
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on opposite HTF trend (after min hold) - adaptive exit
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
            # LONG: Breakout above + oversold momentum + volume confirm + 1d uptrend
            if bullish_breakout and wr_oversold and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Breakdown below + overbought momentum + volume confirm + 1d downtrend
            elif bearish_breakout and wr_overbought and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals