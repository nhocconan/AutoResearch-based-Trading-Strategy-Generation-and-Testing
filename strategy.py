#!/usr/bin/env python3
"""
Experiment #011: 6h Primary + 1w/1d HTF — Volatility Squeeze Breakout + Multi-TF HMA

Hypothesis: After 10 failed experiments using CRSI/CHOP/CPR/pivot patterns, try a 
DIFFERENT approach: Volatility contraction/expansion cycle with multi-TF trend confirmation.

Key insights from research:
- Volatility mean reversion: ATR(7)/ATR(30) extremes predict reversals
- BB squeeze breakouts have 60-70% win rate when aligned with HTF trend
- 6h is unexplored timeframe - middle ground between 4h (too noisy) and 12h (too slow)
- Use 1w HMA for major trend bias (looser filter = more trades)
- Use 1d HMA for intermediate confirmation

Why this differs from failed experiments:
- NO CRSI (failed in #002, #006, #007, #010)
- NO CHOP regime switching (failed in #002, #004, #006, #007, #008)
- NO CPR patterns (failed in #005, #009)
- NO pivot-based entries (failed in #003)
- Uses VOLATILITY CYCLE instead (never tried on 6h)

Design:
- Timeframe: 6h (target 30-60 trades/year)
- HTF: 1w HMA50 for major trend, 1d HMA21 for intermediate
- Entry: BB squeeze (width < 20th percentile) + Donchian breakout + RSI filter
- Exit: 2.5x ATR trailing stoploss
- Size: 0.28 (28% of capital)

Target: Sharpe>0.02, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_squeeze_hma_1w1d_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # normalized width
    
    return upper, lower, width

def calculate_donchian(high, low, period=20):
    """Donchian Channel"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_vol_ratio(atr_short, atr_long):
    """Volatility ratio: ATR(7)/ATR(30) - low = squeeze, high = expansion"""
    n = len(atr_short)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(n):
        if np.isnan(atr_short[i]) or np.isnan(atr_long[i]) or atr_long[i] < 1e-10:
            continue
        ratio[i] = atr_short[i] / atr_long[i]
    
    return ratio

def calculate_percentile_rank(values, lookback=100):
    """Percentile rank of current value over lookback period"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(lookback, n):
        window = values[i-lookback:i+1]
        if np.isnan(window).any():
            continue
        current = values[i]
        pr[i] = np.sum(window < current) / len(window) * 100.0
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_mult=2.0)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    vol_ratio = calculate_vol_ratio(atr_7, atr_30)
    vol_ratio_pr = calculate_percentile_rank(vol_ratio, lookback=100)
    
    # BB width percentile (low = squeeze)
    bb_width_pr = calculate_percentile_rank(bb_width, lookback=100)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need enough data for percentile calculations
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(bb_width[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_ratio_pr[i]) or np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w and 1d HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both agree
        strong_bull = htf_1w_bull and htf_1d_bull
        strong_bear = htf_1w_bear and htf_1d_bear
        
        # === VOLATILITY REGIME ===
        # vol_ratio_pr < 20 = volatility at low percentile (squeeze)
        # vol_ratio_pr > 80 = volatility at high percentile (expansion)
        vol_squeeze = vol_ratio_pr[i] < 25.0
        vol_expansion = vol_ratio_pr[i] > 75.0
        
        # BB width squeeze (width at low percentile)
        bb_squeeze = bb_width_pr[i] < 20.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1]
        
        # === RSI FILTER (moderate - avoid extremes) ===
        rsi_neutral_long = 30.0 < rsi[i] < 70.0
        rsi_neutral_short = 30.0 < rsi[i] < 70.0
        rsi_ok_long = rsi[i] > 25.0
        rsi_ok_short = rsi[i] < 75.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === DESIRED SIGNAL (Vol Squeeze + Breakout + HTF) ===
        desired_signal = 0.0
        
        # LONG: squeeze + breakout + HTF bull + RSI ok
        if vol_squeeze and donchian_breakout_bull and strong_bull and rsi_ok_long:
            desired_signal = SIZE
        elif vol_squeeze and donchian_breakout_bull and hma_bull and rsi_neutral_long:
            desired_signal = SIZE * 0.7
        elif bb_squeeze and donchian_breakout_bull and htf_1w_bull and rsi[i] > 35.0:
            desired_signal = SIZE * 0.7
        # Fallback: HTF strong bull + HMA bull + RSI neutral
        elif strong_bull and hma_bull and rsi_neutral_long and vol_ratio[i] < 1.5:
            desired_signal = SIZE * 0.5
        
        # SHORT: squeeze + breakout + HTF bear + RSI ok
        if desired_signal == 0.0:
            if vol_squeeze and donchian_breakout_bear and strong_bear and rsi_ok_short:
                desired_signal = -SIZE
            elif vol_squeeze and donchian_breakout_bear and hma_bear and rsi_neutral_short:
                desired_signal = -SIZE * 0.7
            elif bb_squeeze and donchian_breakout_bear and htf_1w_bear and rsi[i] < 65.0:
                desired_signal = -SIZE * 0.7
            # Fallback: HTF strong bear + HMA bear + RSI neutral
            elif strong_bear and hma_bear and rsi_neutral_short and vol_ratio[i] < 1.5:
                desired_signal = -SIZE * 0.5
        
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
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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