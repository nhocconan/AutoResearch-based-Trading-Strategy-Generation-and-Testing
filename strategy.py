#!/usr/bin/env python3
"""
Experiment #027: 4h TRIX Momentum + 1d EMA50 + Volume Spike + CHOP Regime

HYPOTHESIS: TRIX(12) smooths noise better than RSI/MACD while catching real momentum shifts.
Combined with 1d EMA50 for trend (proven in #003 with Sharpe 0.123), this should work
in BOTH bull (2021, 2024-2025) and bear (2022):
- Bull: TRIX crosses positive + price > 1d EMA50 = momentum continuation
- Bear: TRIX crosses negative + price < 1d EMA50 = short rallies
- Range: CHOP > 61 = stay out (avoid whipsaw in 2022)

ENTRY: TRIX turns positive + Close > EMA50 + Volume > 1.5x MA(20) + CHOP < 55
EXIT: Opposite TRIX signal or ATR 2.5x stoploss
TARGET: 80-150 total over 4 years (20-37/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_momentum_1d_ema_vol_v1"
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

def calculate_trix(close, period=12):
    """
    TRIX(12) - Triple EMA momentum oscillator
    TRIX > 0 = bullish momentum, TRIX < 0 = bearish momentum
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA (momentum)
    trix = np.full(n, np.nan)
    for i in range(period * 3, n):
        if ema3[i-1] != 0 and not np.isnan(ema3[i-1]):
            trix[i] = ((ema3[i] - ema3[i-1]) / ema3[i-1]) * 100
    
    return trix

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
        
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        hl_range = highest_high - lowest_low
        
        if hl_range > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / hl_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA50 for trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix_12 = calculate_trix(close, period=12)
    trix_prev = np.roll(trix_12, 1)
    trix_prev[0] = np.nan
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Need enough for TRIX triple EMA (period * 3 = 36)
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix_12[i]) or np.isnan(chop_14[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(trix_prev[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME CHECK: Avoid choppy markets ===
        is_trending = chop_14[i] < 55.0  # Don't trade when CHOP > 55
        
        # === TRIX MOMENTUM SHIFT ===
        trix_turning_up = (trix_prev[i] < 0) and (trix_12[i] > 0)
        trix_turning_down = (trix_prev[i] > 0) and (trix_12[i] < 0)
        
        # === TREND DIRECTION FROM 1d EMA ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
        # === TRAILING STOP UPDATE ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 4 bars (16h) to avoid immediate whipsaw ===
        min_hold = (i - entry_bar) >= 4
        
        # === STOPLOSS CHECK (ATR trailing) ===
        stop_hit = False
        if in_position:
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on TRIX reversal (after min hold) OR trend change
            if min_hold:
                if position_side > 0 and (trix_12[i] < 0 or htf_bearish):
                    stop_hit = True
                if position_side < 0 and (trix_12[i] > 0 or htf_bullish):
                    stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # === NEW POSITIONS ===
        # Long: TRIX turns positive + above 1d EMA + volume spike + trending regime
        if is_trending and trix_turning_up and htf_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short: TRIX turns negative + below 1d EMA + volume spike + trending regime
        elif is_trending and trix_turning_down and htf_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals