#!/usr/bin/env python3
"""
Experiment #024: 12h TRIX Momentum Crossover + 1w Trend + Volume

HYPOTHESIS: TRIX(15) crossing zero captures mid-cycle momentum reversals 
on the 12h timeframe. Combined with 1w HMA trend filter (bias only, no reversal) 
and volume confirmation, this should catch institutional momentum shifts while 
avoiding the overtrading that plagued Alligator strategies (267 trades vs 200 max).

KEY INSIGHT: TRIX crossover triggers more frequently than Donchian breakout 
on 12h (which generates 0 trades), but less than Williams Alligator wakeup 
(which generated 267). Target: 75-150 total trades over 4 years.

TIMEFRAME: 12h primary | HTF: 1w for trend bias
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_vol_1w_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=15):
    """TRIX - Triple EMA Rate of Change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Rate of change of triple EMA
    trix = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] != 0:
            trix[i] = ((ema3[i] - ema3[i - 1]) / ema3[i - 1]) * 100
    
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

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend bias (bull/bear filter)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    trix_15 = calculate_trix(close, period=15)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix_15).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix_15[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === 1w TREND FILTER ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # === TRIX CROSSOVER DETECTION ===
        trix_val = trix_15[i]
        trix_sig = trix_signal[i]
        trix_prev = trix_15[i - 1] if i > 0 else 0.0
        trix_sig_prev = trix_signal[i - 1] if i > 0 else 0.0
        
        # Bullish crossover: TRIX crosses above signal
        bullish_cross = (trix_val > trix_sig) and (trix_prev <= trix_sig_prev)
        # Bearish crossover: TRIX crosses below signal
        bearish_cross = (trix_val < trix_sig) and (trix_prev >= trix_sig_prev)
        
        # TRIX zero line cross (momentum shift)
        trix_above_zero = trix_val > 0
        trix_prev_above_zero = trix_prev > 0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === RSI MOMENTUM ===
        rsi_val = rsi[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Bullish TRIX cross + bullish 1w trend + volume
            if bullish_cross and price_above_1w_hma and vol_spike:
                desired_signal = SIZE
            # Alternative: TRIX crosses above zero with RSI momentum
            elif trix_above_zero and not trix_prev_above_zero and price_above_1w_hma:
                if rsi_val > 50:
                    desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Bearish TRIX cross + bearish 1w trend + volume
            if bearish_cross and not price_above_1w_hma and vol_spike:
                desired_signal = -SIZE
            # Alternative: TRIX crosses below zero with RSI bearish
            elif not trix_above_zero and trix_prev_above_zero and not price_above_1w_hma:
                if rsi_val < 50:
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
        
        # === EXIT LOGIC ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: opposite signal or TRIX turns negative
            if bearish_cross:
                exit_triggered = True
            if rsi_val < 35:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: opposite signal or TRIX turns positive
            if bullish_cross:
                exit_triggered = True
            if rsi_val > 65:
                exit_triggered = True
        
        if exit_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
            # Same direction: maintain signal
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals