#!/usr/bin/env python3
"""
Experiment #025: 4h Donchian(20) Breakout + Volume + 1d EMA Trend

HYPOTHESIS: 4h is the proven sweet spot from 16K+ experiments.
Donchian(20) on 4h = ~80 bars/week = natural trade frequency.
Volume confirmation filters false breakouts.
1d EMA filter keeps us aligned with macro trend.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Price breaks above Donchian high = trend continuation, ride up
- Bear: Price breaks below Donchian low = short rally continuation
- Range: Fewer breakouts = adaptive (waits for real moves)

ENTRY: Close > Donchian High(20) AND Volume > 1.3x MA(20)
SHORT: Close < Donchian Low(20) AND Volume > 1.3x MA(20)
FILTER: 1d EMA direction (bull market only go long, bear market only go short)
EXIT: ATR trailing stop (2.5x) or opposite signal

TARGET: 100-200 total over 4 years (25-50/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_vol_1d_ema_v2"
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

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20 (shift by 1 to avoid look-ahead)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
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
    
    warmup = 50
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND FILTER ===
        # Only go long in uptrend, only go short in downtrend
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        bullish_breakout = (close[i] > dc_upper_20[i]) if not np.isnan(dc_upper_20[i]) else False
        bearish_breakout = (close[i] < dc_lower_20[i]) if not np.isnan(dc_lower_20[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > vol_ma_20[i] * 1.3 if vol_ma_20[i] > 1e-10 else False
        
        # === TRAILING STOP UPDATE ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === MIN HOLD: 2 bars (8h) to avoid immediate whipsaw ===
        min_hold = (i - entry_bar) >= 2
        
        # === STOPLOSS CHECK (ATR trailing) ===
        stop_hit = False
        if in_position:
            if position_side > 0:
                stop_hit = low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                stop_hit = high[i] > (lowest_since_entry + 2.5 * atr_14[i])
            
            # Exit on opposite trend signal (after min hold)
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
            continue
        
        # === NEW POSITIONS ===
        # Long: Bullish breakout + volume confirm + 1d uptrend
        if bullish_breakout and vol_ok and htf_bullish:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        
        # Short: Bearish breakdown + volume confirm + 1d downtrend
        elif bearish_breakout and vol_ok and htf_bearish:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        
        else:
            signals[i] = 0.0
    
    return signals