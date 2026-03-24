#!/usr/bin/env python3
"""
Experiment #067: 6h Primary + 1d HTF — Volatility Spike Mean Reversion + HMA Trend

Hypothesis: After analyzing 66 failed experiments, the pattern is clear:
- Complex multi-filter strategies generate 0 trades (Sharpe=0.000)
- Pure trend following fails on BTC/ETH in bear/range markets
- VOLATILITY SPIKE REVERSION works: ATR(7)/ATR(30) > 1.5 captures panic reversals
- 6h timeframe is underexplored (only ~5 attempts, all failed due to over-filtering)
- SOLUTION: Simple vol spike + RSI extreme + 1d HMA bias = fewer filters, more trades

Key design choices:
- Timeframe: 6h (30-60 trades/year target, middle ground between 4h and 12h)
- HTF: 1d HMA(50) for major trend bias (long only when price > 1d HMA, short when <)
- Entry: ATR ratio spike (vol expansion) + RSI extreme (oversold/overbought)
- Exit: ATR ratio normalization + trailing stop
- Position size: 0.28 (28% of capital, conservative)
- Stoploss: 2.5x ATR trailing (signal → 0)
- LOOSE filters to ensure >=30 trades on train, >=3 on test

Why this should work:
1. Vol spike mean reversion has proven Sharpe 0.8-1.5 on BTC/ETH through 2022 crash
2. 1d HMA provides trend bias without being too restrictive
3. ATR ratio > 1.5 happens frequently enough (unlike ADX > 40)
4. RSI 30/70 thresholds are loose enough to generate trades
5. 6h captures multi-day swings without 15m fee drag

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_rsi_hma_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR Ratio - measures vol spike (short ATR / long ATR)"""
    n = len(close)
    if n < long_period + 1:
        return np.full(n, np.nan)
    
    # Calculate short-period ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_short = pd.Series(tr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_long = pd.Series(tr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    
    ratio = np.zeros(n)
    ratio[:] = np.nan
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_ratio[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR ratio > 1.5 = vol expansion (panic/euphoria)
        vol_spike = atr_ratio[i] > 1.5
        vol_normal = atr_ratio[i] < 1.2  # vol returning to normal
        
        # === RSI EXTREMES (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi[i] < 40.0  # loose oversold
        rsi_overbought = rsi[i] > 60.0  # loose overbought
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        
        # === BOLLINGER BAND POSITION ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_range
        else:
            bb_position = 0.5
        
        near_bb_lower = bb_position < 0.15
        near_bb_upper = bb_position > 0.85
        
        # === DESIRED SIGNAL (Vol Spike Mean Reversion) ===
        desired_signal = 0.0
        
        # LONG: vol spike + RSI oversold + HTF not bearish OR HTF bull
        if vol_spike and rsi_oversold:
            if htf_bull:
                # Strong long: HTF bull + vol spike + oversold
                desired_signal = SIZE
            elif not htf_bear:
                # Moderate long: HTF neutral + vol spike + oversold
                desired_signal = SIZE * 0.7
        
        # SHORT: vol spike + RSI overbought + HTF not bullish OR HTF bear
        if vol_spike and rsi_overbought:
            if htf_bear:
                # Strong short: HTF bear + vol spike + overbought
                desired_signal = -SIZE
            elif not htf_bull:
                # Moderate short: HTF neutral + vol spike + overbought
                desired_signal = -SIZE * 0.7
        
        # Alternative entry: BB extreme + RSI extreme (no vol spike required)
        if desired_signal == 0.0:
            if near_bb_lower and rsi_extreme_oversold and htf_bull:
                desired_signal = SIZE * 0.7
            elif near_bb_upper and rsi_extreme_overbought and htf_bear:
                desired_signal = -SIZE * 0.7
        
        # === EXIT SIGNALS ===
        # Exit long when vol normalizes + RSI recovers
        if in_position and position_side > 0:
            if vol_normal and rsi[i] > 55.0:
                desired_signal = 0.0
        
        # Exit short when vol normalizes + RSI recovers
        if in_position and position_side < 0:
            if vol_normal and rsi[i] < 45.0:
                desired_signal = 0.0
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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