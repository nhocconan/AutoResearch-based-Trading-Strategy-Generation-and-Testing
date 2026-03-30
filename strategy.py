#!/usr/bin/env python3
"""
Experiment #006: 4h TRIX Momentum + Volume Spike + ATR Regime

HYPOTHESIS: TRIX (Triple EMA Oscillator) smooths market noise better than 
single/double EMA. When TRIX crosses its signal line WITH volume confirmation,
it catches institutional momentum shifts. ATR regime filter avoids false 
signals in low-volatility choppy markets.

WHY IT WORKS IN BULL AND BEAR: TRIX catches momentum in BOTH directions.
Long when TRIX crosses up + vol spike in uptrend. Short when TRIX crosses down
+ vol spike in downtrend. Symmetric logic works in all markets.

TIMELINE:
- Train: 2021-2024 (4 years, ~3500 4h bars)
- Test: 2025-2026 (15 months, ~2700 4h bars)

TARGET: 75-200 total trades over 4 years = 19-50/year.
TRIX crossover is tighter than EMA crossover = fewer but higher quality signals.
Signal size: 0.25-0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_vol_atrregime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=14, signal=9):
    """
    Triple EMA Oscillator (TRIX)
    TRIX = Rate of change of triple EMA (EMA of EMA of EMA)
    Signal line = EMA of TRIX
    """
    n = len(close)
    if n < period * 3 + signal:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # TRIX = 100 * rate of change of triple EMA
    trix = np.zeros(n)
    for i in range(1, n):
        if ema3.iloc[i-1] != 0:
            trix[i] = 100 * (ema3.iloc[i] - ema3.iloc[i-1]) / ema3.iloc[i-1]
    
    # Signal line = EMA of TRIX
    trix_series = pd.Series(trix)
    signal_line = trix_series.ewm(span=signal, min_periods=signal, adjust=False).mean().values
    
    return trix, signal_line

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
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix, signal_line = calculate_trix(close, period=14, signal=9)
    
    # Volume ratio (vol spike detection)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR regime: compare current ATR to 30-bar average
    # ATR at 30d low = low volatility (chop), ATR at 30d high = high volatility (trending)
    atr_ma = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
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
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix[i]) or np.isnan(signal_line[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        price_below_1d_ema = close[i] < ema_1d_aligned[i]
        
        # === TRIX CROSSOVERS ===
        # Bullish: TRIX crosses above signal line
        trix_cross_up = trix[i] > signal_line[i] and trix[i-1] <= signal_line[i-1]
        # Bearish: TRIX crosses below signal line
        trix_cross_down = trix[i] < signal_line[i] and trix[i-1] >= signal_line[i-1]
        
        # TRIX momentum confirmation: TRIX and signal both positive/negative
        trix_bullish = trix[i] > 0 and signal_line[i] > 0
        trix_bearish = trix[i] < 0 and signal_line[i] < 0
        
        # Volume confirmation: volume spike > 1.5x average
        vol_spike = vol_ratio[i] > 1.5
        
        # ATR regime: only enter when ATR ratio > 1.0 (not in chop)
        # This avoids false signals during low-volatility consolidation
        atr_expanding = atr_ratio[i] > 1.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Conditions: TRIX cross up + volume spike + ATR expanding + above 1d EMA
            if price_above_1d_ema and trix_cross_up and vol_spike and atr_expanding:
                desired_signal = SIZE
            
            # Alternative: TRIX momentum confirmation with strong volume
            elif price_above_1d_ema and trix_bullish and vol_ratio[i] > 2.0:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Conditions: TRIX cross down + volume spike + ATR expanding + below 1d EMA
            if price_below_1d_ema and trix_cross_down and vol_spike and atr_expanding:
                desired_signal = -SIZE
            
            # Alternative: TRIX momentum confirmation with strong volume
            elif price_below_1d_ema and trix_bearish and vol_ratio[i] > 2.0:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
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
        
        # === TAKE PROFIT: 3R or TRIX reversal ===
        bars_held = i - entry_bar
        
        if in_position:
            pnl_r = 0.0
            if position_side > 0:
                pnl_r = (close[i] - entry_price) / entry_atr
            else:
                pnl_r = (entry_price - close[i]) / entry_atr
            
            # Take profit at 3R with trailing stop
            if pnl_r >= 3.0:
                desired_signal = SIZE / 2  # Half position
            
            # Exit if TRIX reverses
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