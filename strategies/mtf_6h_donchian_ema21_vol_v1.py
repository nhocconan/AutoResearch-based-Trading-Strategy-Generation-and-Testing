#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian Breakout + Daily EMA21 Trend + Volume (6h)

HYPOTHESIS: Simplify from 4 conditions to 3 core conditions:
1. Daily EMA21 for trend direction (less lag than weekly VWAP)
2. Donchian(20) breakout for entry signal
3. Volume spike for confirmation

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Price above EMA21 + breakout above 20-bar high + volume spike = strong momentum
- Bear: Price below EMA21 + breakdown below 20-bar low + volume spike = strong short
- 2.5 ATR stop allows surviving pullbacks; EMA21 exit catches trend reversals

EXPECTED TRADES: 200-400 total over 4 years (50-100/year per symbol)
- Donchian(20) on 6h = 1 breakout per 20-40 bars = ~200-400/year potential
- Volume spike (1.5x) → reduces by ~40%
- EMA21 trend filter → reduces by ~30%
- Final: ~300-500 total over 4 years (needs to verify not overtrading)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_ema21_vol_v1"
timeframe = "6h"
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
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA21 for trend (align to 6h)
    daily_ema21 = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_aligned = align_htf_to_ltf(prices, df_1d, daily_ema21)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Enough for Donchian20, ATR14, EMA21 alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: Daily EMA21 ===
        bull_trend = close[i] > ema21_aligned[i]
        bear_trend = close[i] < ema21_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        min_hold_passed = (i - entry_bar) >= 3 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Update trailing high/low
            if position_side > 0:
                if high[i] > trailing_high:
                    trailing_high = high[i]
            else:
                if low[i] < trailing_low:
                    trailing_low = low[i]
            
            # Stop-loss: 2.5 ATR from entry
            stop_price = entry_price - 2.5 * entry_atr if position_side > 0 else entry_price + 2.5 * entry_atr
            stop_hit = (position_side > 0 and low[i] < stop_price) or (position_side < 0 and high[i] > stop_price)
            
            # Trend exit
            trend_exit = (position_side > 0 and close[i] < ema21_aligned[i]) or \
                        (position_side < 0 and close[i] > ema21_aligned[i])
            
            if stop_hit or (min_hold_passed and trend_exit):
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend
            if bullish_breakout and vol_spike and bull_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend
            elif bearish_breakout and vol_spike and bear_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_low = low[i]
                signals[i] = -SIZE
    
    return signals