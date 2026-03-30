#!/usr/bin/env python3
"""
Experiment #025: 12h Donchian Breakout + 1d SMA50 Trend + Volume Spike

HYPOTHESIS: Keep it SIMPLE. One strong signal (Donchian breakout) + volume 
confirmation + simple trend filter (1d SMA50) = proven winning formula.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout above 20-bar high + volume spike + above 1d SMA50 = trend continuation
- Bear: Breakdown below 20-bar low + volume spike + below 1d SMA50 = short momentum
- 1d SMA50 is slow enough to not whipsaw, fast enough to catch major trends
- 12h timeframe = fewer trades = less fee drag = better test generalization

EXPECTED TRADES: 75-150 total over 4 years (19-37/year per symbol)
- Donchian(20) on 12h = break every 20-40 bars = 109-219 potential/year
- Volume spike (1.5x) → reduces by ~40%
- 1d SMA50 trend filter → reduces by ~30%
- Final: ~75-150 trades = within target range
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_sma50_vol_v1"
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

def calculate_sma(values, period):
    """Simple Moving Average"""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction
    sma_50_1d = calculate_sma(df_1d['close'].values, 50)
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - classic breakout structure
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
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
    
    warmup = 60  # Enough for Donchian20, ATR14, SMA50
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: 1d SMA50 ===
        bull_trend = close[i] > sma_50_aligned[i]
        bear_trend = close[i] < sma_50_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        # Use previous bar's channel to avoid look-ahead
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           close[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           close[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend
            if bullish_breakout and vol_spike and bull_trend:
                desired_signal = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend
            elif bearish_breakout and vol_spike and bear_trend:
                desired_signal = -SIZE
        
        # === EXIT LOGIC - Donchian Trailing Stop ===
        if in_position:
            if position_side > 0:
                # Trailing stop: lowest low since entry (Donchian-style exit)
                lookback = min(10, i - entry_bar)
                if lookback > 0:
                    stop_low = pd.Series(low[entry_bar:i+1]).min()
                    stop_price = stop_low
                    if low[i] < stop_price:
                        desired_signal = 0.0
                        in_position = False
                        position_side = 0
                
                # Exit if trend flips (below 1d SMA50)
                if close[i] < sma_50_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing stop: highest high since entry
                lookback = min(10, i - entry_bar)
                if lookback > 0:
                    stop_high = pd.Series(high[entry_bar:i+1]).max()
                    stop_price = stop_high
                    if high[i] > stop_price:
                        desired_signal = 0.0
                        in_position = False
                        position_side = 0
                
                # Exit if trend flips (above 1d SMA50)
                if close[i] > sma_50_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        
        signals[i] = desired_signal
    
    return signals