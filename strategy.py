#!/usr/bin/env python3
"""
Experiment #003: 4h Donchian Breakout + Volume Spike + 1d Trend

HYPOTHESIS: Simple Donchian(20) breakouts with volume confirmation and 1d trend
filter will work in both bull and bear markets because:
- Donchian channels are derived from price structure, not direction
- Bull markets: breakouts above channel = long
- Bear markets: breakouts below channel = short
- Volume spike confirms institutional participation
- 1d trend filter prevents counter-trend trades

KEY DESIGN (proven pattern from DB):
1. Donchian(20) breakout (HH/LL in 20 bars)
2. Volume spike (>1.8x 20-avg) confirming breakout
3. 1d HMA(21) for trend direction (bull: price>1d HMA, bear: price<1d HMA)
4. ATR(14) stoploss (2x ATR)
5. Simple AND logic — all conditions must agree

TARGET: 75-150 total trades over 4 years (proven ~50 trades on 4h is minimum).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95 trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_hma_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper = highest high, lower = lowest low"""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    mid = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === HTF DATA: Load 1d for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # === LTF INDICATORS ===
    # Donchian(20) channels
    dc_upper, dc_lower, dc_mid = calculate_donchian(high, low, period=20)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume moving average (20 bars = 80h = 3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === SIGNAL GENERATION ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_in_trade = 0
    
    # Cooldown: minimum bars between trades (120 bars = 20 days to avoid whipsaw)
    cooldown_bars = 120
    bars_since_exit = cooldown_bars  # Start ready to trade
    
    # Warmup: need at least 60 bars for all indicators
    warmup = 60
    
    for i in range(warmup, n):
        # Check if indicators are ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]) or vol_ratio[i] <= 0:
            signals[i] = 0.0
            continue
        
        # === 1d TREND FILTER ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout UP: close above 20-bar high
        breakout_up = close[i] > dc_upper[i - 1] if not np.isnan(dc_upper[i - 1]) else False
        
        # Breakout DOWN: close below 20-bar low
        breakout_down = close[i] < dc_lower[i - 1] if not np.isnan(dc_lower[i - 1]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        bars_since_exit += 1
        
        # Update bars in trade
        if in_position:
            bars_in_trade += 1
        
        # === STOPLOSS CHECK ===
        stoploss_triggered = False
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                stoploss_triggered = True
        
        # === TAKE PROFIT: Exit after 4 ATR move ===
        tp_triggered = False
        if in_position:
            if position_side > 0:
                profit = (high[i] - entry_price) / entry_atr
                if profit >= 4.0:
                    tp_triggered = True
            if position_side < 0:
                profit = (entry_price - low[i]) / entry_atr
                if profit >= 4.0:
                    tp_triggered = True
        
        # === COOLDOWN EXIT ===
        # Force exit after holding too long without profit
        if in_position and bars_in_trade > 60:
            if position_side > 0:
                profit_pct = (close[i] - entry_price) / entry_price
                if profit_pct < 0.02:  # Less than 2% profit after 10 days
                    desired_signal = 0.0  # Exit
        
        if stoploss_triggered or tp_triggered:
            desired_signal = 0.0
            bars_since_exit = 0
        
        # === ENTRY CONDITIONS ===
        # ONLY enter if ALL conditions agree (strict AND logic)
        # Cooldown check
        if bars_since_exit < cooldown_bars:
            desired_signal = 0.0
        
        # LONG: Breakout up + volume spike + trend bullish
        if not stoploss_triggered and not tp_triggered and desired_signal == 0.0:
            if breakout_up and vol_spike and price_above_1d_hma:
                desired_signal = SIZE
            
            # SHORT: Breakout down + volume spike + trend bearish
            if breakout_down and vol_spike and not price_above_1d_hma:
                desired_signal = -SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_in_trade = 0
                bars_since_exit = 0
        else:
            # Exit
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = desired_signal
    
    return signals