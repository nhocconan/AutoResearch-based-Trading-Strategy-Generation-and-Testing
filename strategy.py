#!/usr/bin/env python3
"""
Experiment #006: 4h TRIX + Donchian Breakout + Volume + 1d EMA200 Trend

HYPOTHESIS: TRIX is a smoothed momentum oscillator that filters noise better than
raw EMA crossovers. Donchian(20) provides clear structural breakouts. Volume
confirms institutional involvement. 1d EMA200 filters for major trend direction.

This combination targets 75-150 total trades (19-37/year) with high confluence.
TRIX crossover is slower than EMA = fewer but higher quality signals.

WHY IT WORKS IN BULL AND BEAR:
- Bull: TRIX crosses positive + price breaks Donchian high + above 1d EMA200
- Bear: TRIX crosses negative + price breaks Donchian low + below 1d EMA200
- Reversals: TRIX divergence at Donchian boundaries

TARGET: 75-150 total over 4 years. HARD MAX: 300.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_donchian_vol_1d_v2"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=9):
    """TRIX indicator: triple smoothed rate of change"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA smoothing
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA
    trix = np.zeros(n, dtype=np.float64)
    trix[period*3:] = (ema3.values[period*3:] - ema3.values[period*3 - period: -period]) / ema3.values[period*3 - period: -period] * 100
    
    return trix

def calculate_donchian(high, low, period=20):
    """Donchian Channel: returns upper, lower, middle"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    trix = calculate_trix(close, period=9)
    trix_signal = pd.Series(trix).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume metrics
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 300  # Need enough for EMA200 alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Previous bar TRIX for crossover detection
        trix_prev = trix[i - 1]
        trix_curr = trix[i]
        signal_prev = trix_signal[i - 1]
        signal_curr = trix_signal[i]
        
        # TRIX crossover: positive = bullish, negative = bearish
        trix_cross_up = (trix_prev < signal_prev) and (trix_curr >= signal_curr)
        trix_cross_down = (trix_prev > signal_prev) and (trix_curr <= signal_curr)
        
        # 1d trend direction
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout signals
        donchian_break_high = high[i] > donchian_upper[i - 1]  # Break above previous upper
        donchian_break_low = low[i] < donchian_lower[i - 1]    # Break below previous lower
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX crosses up + Donchian breakout + above 1d EMA + volume ===
            if trix_cross_up and donchian_break_high and price_above_1d_ema and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TRIX crosses down + Donchian breakout + below 1d EMA + volume ===
            if trix_cross_down and donchian_break_low and price_below_1d_ema and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === EXIT: TRIX reversal ===
        if in_position:
            if position_side > 0 and trix_cross_down:
                desired_signal = 0.0
            if position_side < 0 and trix_cross_up:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
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
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals