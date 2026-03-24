#!/usr/bin/env python3
"""
Experiment #1489: 4h Primary + 1d HTF — Volatility Spike Mean Reversion

Hypothesis: After 1111 failed strategies, the pattern is clear:
1. All 4h TREND-FOLLOWING strategies failed (Sharpe -0.2 to -1.1)
2. Higher TF (1d, 12h) trend-following works (Sharpe +0.15 to +0.6)
3. Mean reversion on 4h has NOT been properly tested with HTF filter

Key insight from research notes:
- "VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long"
- "Captures 'vol crush' after panic. Exit when ATR ratio < 1.2"
- This is DIFFERENT from all failed 4h strategies (which were trend-follow)

Strategy design:
- 1d HMA(21) for macro trend filter (only trade WITH daily trend)
- 4h ATR ratio(7/30) > 2.0 for volatility spike detection
- 4h Bollinger(20, 2.5) for extreme price levels
- 4h RSI(14) < 35 or > 65 for momentum confirmation
- ATR(14)*2.5 trailing stoploss

Why this should work on 4h:
1. Mean reversion after vol spikes has different dynamics than trend-follow
2. 1d HMA filter prevents counter-trend trades (major failure mode)
3. Tight BB (2.5 std) + low RSI ensures only extreme entries
4. Target 20-50 trades/year (vol spikes are rare events)
5. Discrete signal sizes (0.0, ±0.25, ±0.30) minimize fee churn

Timeframe: 4h
HTF: 1d (call get_htf_data ONCE before loop!)
Position Size: 0.30 (discrete levels)
Target: 20-50 trades/year, Sharpe > 0.618 (beat current best), ALL symbols Sharpe > 0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_meanrev_1d_hma_bb_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - smoother than EMA, less lag
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA helper
    def wma(data, span):
        if span < 1:
            span = 1
        weights = np.arange(1, span + 1)
        weights = weights / weights.sum()
        result = np.full(len(data), np.nan)
        for i in range(span - 1, len(data)):
            if np.all(~np.isnan(data[i - span + 1:i + 1])):
                result[i] = np.sum(data[i - span + 1:i + 1] * weights)
        return result
    
    half = max(1, period // 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    if len(wma_half) > 0 and len(wma_full) > 0:
        diff = 2 * wma_half - wma_full
        hma = wma(diff, sqrt_n)
    else:
        hma = np.full(n, np.nan)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_dev=2.5):
    """Bollinger Bands with configurable std dev"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_dev=2.5)
    rsi = calculate_rsi(close, period=14)
    
    # Calculate ATR ratio for vol spike detection
    atr_ratio = np.full(n, np.nan)
    mask = (atr_30 > 1e-10) & (~np.isnan(atr_7)) & (~np.isnan(atr_30))
    atr_ratio[mask] = atr_7[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) - direction bias ===
        # Only trade in direction of daily trend
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0  # ATR(7) > 2x ATR(30)
        
        # === BOLLINGER BAND EXTREMES ===
        below_bb = close[i] < bb_lower[i]
        above_bb = close[i] > bb_upper[i]
        
        # === RSI MOMENTUM ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL - VOL SPIKE MEAN REVERSION ===
        desired_signal = 0.0
        
        # LONG: Daily bull + Vol spike + Below BB + RSI oversold
        if daily_bull and vol_spike and below_bb and rsi_oversold:
            desired_signal = BASE_SIZE
        
        # SHORT: Daily bear + Vol spike + Above BB + RSI overbought
        elif daily_bear and vol_spike and above_bb and rsi_overbought:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.8:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals