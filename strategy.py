#!/usr/bin/env python3
"""
Experiment #028: 6h TRIX Momentum + Volatility Expansion + Volume Confirmation

HYPOTHESIS: TRIX is a triple-smoothed momentum oscillator that filters market noise
better than RSI or Williams%R. By combining TRIX zero-line crossovers with:
1. 1d SMA200 for trend direction (filters counter-trend trades)
2. ATR(7)/ATR(30) ratio for volatility expansion regime (confirms institutional moves)
3. Volume spike confirmation (validates momentum)

This creates tight, high-probability entries with clear exit conditions.

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull: TRIX crosses above 0 + price > 1d SMA200 + vol spike = strong long
- Bear: TRIX crosses below 0 + price < 1d SMA200 + vol spike = strong short
- Volatility expansion filters out low-volatility chop
- 6h timeframe = ~100 trades/year (within target range)

TARGET: 75-200 total trades over 4 years. HARD MAX: 300.
Signal size: 0.25 (conservative, max drawdown protection).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_trix_volatility_expansion_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_trix(close, period=14):
    """TRIX - Triple Smoothed Rate of Change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA smoothing
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple smoothed EMA
    trix = np.zeros(n)
    trix[:] = np.nan
    for i in range(period, n):
        if ema3.iloc[i - period] != 0:
            trix[i] = 100 * (ema3.iloc[i] - ema3.iloc[i - period]) / ema3.iloc[i - period]
    
    return trix

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio: short-term volatility vs long-term volatility"""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.full(len(close), np.nan, dtype=np.float64)
    valid = atr_long > 1e-10
    ratio[valid] = atr_short[valid] / atr_long[valid]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    trix = calculate_trix(close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 250  # Need enough for TRIX + SMA200(1d)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TRIX CROSSOVER SIGNALS ===
        # Need previous bar's TRIX for crossover detection
        prev_trix = trix[i - 1] if i > 0 else 0
        curr_trix = trix[i]
        
        # TRIX crossed above zero = bullish momentum
        trix_cross_up = prev_trix < 0 and curr_trix >= 0
        # TRIX crossed below zero = bearish momentum
        trix_cross_down = prev_trix > 0 and curr_trix <= 0
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        price_below_1d_sma = close[i] < sma_1d_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio > 1.3 = expansion) ===
        # Only trade during volatility expansion (institutional moves)
        vol_expansion = atr_ratio[i] > 1.3
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX crosses above 0 + price above 1d SMA200 + vol expansion ===
            if trix_cross_up and price_above_1d_sma:
                # Require BOTH vol expansion AND volume spike for confirmation
                if vol_expansion and vol_spike:
                    desired_signal = SIZE
            
            # === SHORT: TRIX crosses below 0 + price below 1d SMA200 + vol expansion ===
            if trix_cross_down and price_below_1d_sma:
                if vol_expansion and vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TRIX REVERSAL EXIT (if TRIX crosses back, take profit) ===
        if in_position and position_side > 0 and trix_cross_down:
            # Close long if TRIX reverses down
            desired_signal = 0.0
        
        if in_position and position_side < 0 and trix_cross_up:
            # Close short if TRIX reverses up
            desired_signal = 0.0
        
        # === MINIMUM HOLD TIME (4 bars = 1 day) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 4:
            # If TRIX has reversed AND we have some profit, consider exit
            if position_side > 0 and curr_trix < 0 and close[i] > entry_price:
                desired_signal = 0.0
            if position_side < 0 and curr_trix > 0 and close[i] < entry_price:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals