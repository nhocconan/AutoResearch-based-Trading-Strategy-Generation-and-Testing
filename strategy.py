#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian(16) + 1w EMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: 12h timeframe with weekly HTF filter should work because:
- Weekly EMA gives strong macro trend signal (not whipsawed by daily noise)
- 12h Donchian(16) = 8-day channel - captures medium-term swings
- Volume spike confirms institutional participation
- ATR stoploss protects during 2022-type crashes
- 12h timeframe naturally limits trade frequency to target 12-37/year

WHY BOTH MARKETS:
- 2021 bull: Breakout above 8-day high + weekly trend up = ride rallies
- 2022 bear: Weekly trend down = only short breakdowns (no longs)
- 2025 range: Weekly EMA filter keeps flat in chop

ENTRY LOGIC (simple = fewer trades = less fee drag):
- LONG: Close > Donchian_upper[15] + volume > 1.8x avg + weekly trend UP
- SHORT: Close < Donchian_lower[15] + volume > 1.8x avg + weekly trend DOWN

EXIT LOGIC:
- ATR trailing stop (2.5x ATR from entry high/low)
- Trend reversal (weekly EMA flip)

TARGET: 50-120 total trades over 4 years (12-30/year on 12h).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_021_12h_donchian_1w_ema_vol_v1"
timeframe = "12h"
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
    
    # === HTF: 1w EMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(16) - 8 days on 12h
    donchian_upper = pd.Series(high).rolling(window=16, min_periods=16).max().values
    donchian_lower = pd.Series(low).rolling(window=16, min_periods=16).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume spike detection (20-bar avg on 12h = ~10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
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
    
    warmup = 50  # 16 for Donchian + buffer
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === WEEKLY HTF TREND ===
        weekly_bullish = close[i] > ema_1w_aligned[i]
        weekly_bearish = close[i] < ema_1w_aligned[i]
        
        # === VOLUME SPIKE (>1.8x average) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT (use prior bar's channel for no look-ahead) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = False
        bearish_breakout = False
        
        if not np.isnan(prev_upper) and not np.isnan(prev_lower):
            # Close breaks above prior upper = bullish breakout
            bullish_breakout = close[i] > prev_upper
            # Close breaks below prior lower = bearish breakout
            bearish_breakout = close[i] < prev_lower
        
        # === MINIMUM HOLD: 2 bars (24h) to avoid immediate reversals ===
        min_hold_bars = 2
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ATR TRAILING STOP (2.5x ATR from entry high/low) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Also exit on trend reversal (weekly EMA flip)
            if position_side > 0 and weekly_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and weekly_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume spike + weekly bullish
            if bullish_breakout and vol_spike and weekly_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Bearish breakdown + volume spike + weekly bearish
            elif bearish_breakout and vol_spike and weekly_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals