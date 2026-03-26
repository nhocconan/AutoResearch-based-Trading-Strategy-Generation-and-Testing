#!/usr/bin/env python3
"""
Experiment #008: 12h Donchian Breakout + 1d Trend + Volume Spike

HYPOTHESIS: Simple 20-period Donchian breakout captures institutional moves.
1d HMA trend filter ensures we only trade WITH the larger trend.
Volume spike confirms institutional involvement.
ATR-based stoploss prevents blowups in false breakouts.

WHY THIS WORKS IN BOTH BULL AND BEAR:
- Bull: Price breaks 12h Donchian high + 1d uptrend = strong long
- Bear: Price breaks 12h Donchian low + 1d downtrend = strong short
- Range: No breakout = no trades = no losses

KEY INSIGHT FROM DB:
- Donchian breakout + HMA trend + volume confirmation (SOL Sharpe 1.10-1.38)
- Simple is better: fewer conditions = fewer trades = less fee drag
- Target: 30-60 trades/year on 12h (tight entries required)

RULES:
1. 12h Donchian(20) breakout (price > 20-high OR price < 20-low)
2. 1d HMA(21) confirms trend direction
3. Volume > 1.5x 20-bar average confirms breakout
4. ATR(14) stoploss at 2x
5. Signal: 0.30 (discrete)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_trend_vol_v1"
timeframe = "12h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # =========================================
    # MTF: Load 1d data ONCE for trend
    # =========================================
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # =========================================
    # PRIMARY TF: 12h indicators
    # =========================================
    
    # Donchian channels (20 periods)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    # Warmup - need 20 bars for Donchian
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Skip if 1d trend not ready
        trend_ready = not np.isnan(hma_1d_aligned[i])
        
        # Skip if volume not ready
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # =========================================
        # REGIME CHECK: Only trade if 1d trend aligns
        # =========================================
        bullish_trend = trend_ready and (close[i] > hma_1d_aligned[i])
        bearish_trend = trend_ready and (close[i] < hma_1d_aligned[i])
        
        # =========================================
        # BREAKOUT DETECTION
        # =========================================
        donch_breakout_high = close[i] > donchian_high[i] if not np.isnan(donchian_high[i]) else False
        donch_breakout_low = close[i] < donchian_low[i] if not np.isnan(donchian_low[i]) else False
        
        # =========================================
        # VOLUME CONFIRMATION
        # =========================================
        vol_spike = vol_ratio[i] > 1.5
        
        # =========================================
        # ENTRY LOGIC: Very strict - ALL must agree
        # =========================================
        desired_signal = 0.0
        
        # LONG: Breakout high + bullish 1d trend + volume spike
        if donch_breakout_high and bullish_trend and vol_spike:
            desired_signal = SIZE
        
        # SHORT: Breakout low + bearish 1d trend + volume spike
        if donch_breakout_low and bearish_trend and vol_spike:
            desired_signal = -SIZE
        
        # =========================================
        # STOPLOSS CHECK (ATR-based trailing stop)
        # =========================================
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # =========================================
        # RE-ENTRY: Allow if same direction breakout
        # =========================================
        # Only enter new position if signal changes direction or re-entering same direction after stop
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
            elif in_position and np.sign(desired_signal) == position_side:
                # Already in position - don't change signal
                desired_signal = position_side * SIZE
        else:
            # No entry signal - hold position if in one, otherwise flat
            if in_position:
                # Maintain current position
                desired_signal = position_side * SIZE
            else:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals