#!/usr/bin/env python3
"""
Experiment #009: 4h TRIX Momentum + ATR Regime + 1d EMA Trend

HYPOTHESIS: TRIX momentum crossovers catch trend changes. By combining:
1. 1d EMA50 for trend direction (bull only longs, bear only shorts)
2. TRIX(21) sign change for momentum entry
3. ATR regime (ATR percentile < 20 = squeeze about to break, ADX confirms trend)

WHY IT WORKS: TRIX crossover at extremes often marks trend reversals. ATR regime
filters out choppy markets where TRIX whipsaws. 1d trend alignment ensures we
only trade with the higher timeframe.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_atrregime_ema50_1d_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_trix(prices, period=21):
    """TRIX: triple EMA derivative, removes noise and lag"""
    ema1 = pd.Series(prices).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    trix = ema3.pct_change() * 100
    return trix.values

def calculate_adx(high, low, close, period=14):
    """ADX for regime detection"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr > 0, atr, 1)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix = calculate_trix(close, period=21)
    
    # ATR percentile for squeeze detection
    atr_ma = pd.Series(atr_14).rolling(window=100, min_periods=100).mean().values
    atr_percentile = np.zeros(n)
    for i in range(100, n):
        window = atr_14[i-99:i+1]
        atr_percentile[i] = (atr_14[i] - np.min(window)) / (np.max(window) - np.min(window) + 1e-10)
    
    # ADX for regime detection
    adx = calculate_adx(high, low, close, period=14)
    
    # TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    trix_hist = trix - trix_signal
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    entry_trix = 0.0
    
    warmup = 200  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(ema_1d_aligned[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # === REGIME DETECTION ===
        # Squeeze: ATR percentile below 20 = low volatility, potential breakout
        atr_squeeze = atr_percentile[i] < 20
        # Trending: ADX above 25 = directional market
        adx_trending = adx[i] > 25
        # Range: ADX below 20 = choppy
        adx_range = adx[i] < 20
        
        # Allow trades in trending OR just exited squeeze
        allow_trade = adx_trending or atr_squeeze
        
        # === TRIX MOMENTUM SIGNAL ===
        # Bullish: TRIX crosses above signal (histogram positive)
        trix_bullish = trix_hist[i] > 0 and trix_hist[i-1] <= 0
        # Bearish: TRIX crosses below signal (histogram negative)
        trix_bearish = trix_hist[i] < 0 and trix_hist[i-1] >= 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX bullish cross + trend alignment + regime ===
            if price_above_1d_ema and trix_bullish and allow_trade:
                desired_signal = SIZE
            
            # === SHORT: TRIX bearish cross + trend alignment + regime ===
            if price_below_1d_ema and trix_bearish and allow_trade:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT: TRIX reversal ===
        if in_position:
            bars_held = i - entry_bar
            # Exit if TRIX reverses (momentum fading) AND profitable
            if position_side > 0 and trix_hist[i] < 0 and close[i] > entry_price:
                desired_signal = 0.0
            if position_side < 0 and trix_hist[i] > 0 and close[i] < entry_price:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                entry_trix = trix_hist[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals