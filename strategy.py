#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian Breakout + Volume + Choppiness Regime (12h Trend)

HYPOTHESIS: The proven SOLUSDT pattern (Sharpe 1.49, 107 trades) works because:
1. Donchian(20) breakout catches institutional moves
2. Volume spike confirms smart money involvement
3. Choppiness filter avoids the 2022 crash bottom whipsaw
4. 12h EMA ensures we don't fight macro trend

WHY 4h: Best timeframe per DB evidence (41% keep rate, Sharpe 1.46-1.49).
WHY IT WORKS IN BULL/BEAR: Bull breakouts work, bear short breakouts work.
Choppiness avoids bottom-fishing in ranging/crash markets.

TARGET: 80-150 total trades (20-37/year) — within proven winning range.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_12h_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian channel - upper = highest high, lower = lowest low"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2.0
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - identifies trending vs ranging markets
    CHOP > 61.8 = choppy/ranging (AVOID - use tighter stops or skip)
    CHOP < 38.2 = trending (GOOD - trend following works)
    Formula: 100 * LOG10(SUM(ATR,14) / (HHV(High,14) - LLV(Low,14))) / LOG10(14)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr_j = high[j] - low[j]
            else:
                tr_j = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr_j
        
        # Highest high - lowest low over period
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        range_val = highest_high - lowest_low
        
        if range_val > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h EMA for macro trend (call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)  # auto shift(1)
    
    # === 4h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper, dc_middle, dc_lower = calculate_donchian(high, low, period=20)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume MA for spike detection (30-period for 4h = 5 days)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
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
    
    warmup = 100  # Need 100 bars for all indicators (Chop needs 14, but give buffer)
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(dc_upper[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME CHECK ===
        # Only trade in trending markets (CHOP < 61.8)
        # When CHOP > 61.8, we're in a ranging market - skip new entries
        chop_ok = chop[i] < 61.8
        
        # === TREND DETECTION via 12h EMA ===
        # Use prior bar's EMA for direction (no look-ahead)
        ema_trend_up = close[i] > ema_12h_aligned[i]
        ema_trend_down = close[i] < ema_12h_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long: price closes above 20-bar high + volume spike
        # Use PRIOR bar's upper (shifted by 1) to avoid mid-bar entry
        long_breakout = close[i] > dc_upper[i] if i == 0 else close[i] > dc_upper[i-1]
        # Short: price closes below 20-bar low + volume spike
        short_breakout = close[i] < dc_lower[i] if i == 0 else close[i] < dc_lower[i-1]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 2 bars (8h) to avoid chop ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP (2.5x ATR from highest/lowest since entry) ===
        def check_stop_loss():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: price fell below highest - 2.5*ATR
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                # Short stop: price rose above lowest + 2.5*ATR
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXIT LOGIC ===
        if in_position:
            stop_hit = check_stop_loss()
            
            # Exit if trend reverses AND we've held minimum
            if position_side > 0 and ema_trend_down and min_hold:
                stop_hit = True
            if position_side < 0 and ema_trend_up and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # === NEW POSITIONS (only if not in position AND regime OK) ===
        if not in_position and chop_ok:
            # LONG: Breakout above + volume spike + uptrend
            if long_breakout and vol_spike and ema_trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            
            # SHORT: Breakout below + volume spike + downtrend
            elif short_breakout and vol_spike and ema_trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals