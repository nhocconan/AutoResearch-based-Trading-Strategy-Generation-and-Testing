#!/usr/bin/env python3
"""
Experiment #511: 6h Primary + 1w/1d HTF — Dual HTF Confluence + Fast RSI + BB Regime

Hypothesis: 6h needs BOTH 1w and 1d HTF agreement for stable trend bias.
Single HTF (1d only in #495) was too noisy. Adding 1w filter reduces whipsaws.
Fast RSI(7) instead of RSI(14) for quicker mean reversion signals on 6h.
Bollinger Band width for regime detection (squeeze = prepare, expansion = enter).

Strategy logic:
1. 1w HMA(21) = major trend bias (slowest, most reliable)
2. 1d HMA(21) = intermediate trend confirmation
3. 6h RSI(7) = fast mean reversion entries (25/75 extremes, 35/65 momentum)
4. 6h BB(20,2.0) = volatility regime + mean reversion bounds
5. BB Width percentile = squeeze/expansion detection
6. ATR(14)*2.5 stoploss on all positions
7. Entry requires BOTH 1w AND 1d HTF agreement (confluence)

Key changes from #495:
- Added 1w HTF (dual HTF confluence, not single 1d)
- RSI(7) instead of RSI(14) for faster signals
- Bollinger Bands for regime + mean reversion
- Require HTF confluence (both agree) before entry
- Simpler entry logic (fewer conflicting conditions)

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=12 test
Timeframe: 6h with 1w+1d HTF filters
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_dual_htf_rsi7_bb_1w1d_v1"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index - fast version for 6h"""
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
    
    return upper, sma, lower

def calculate_bb_width(upper, lower, middle):
    """Bollinger Band Width = (Upper - Lower) / Middle"""
    n = len(middle)
    width = np.zeros(n)
    width[:] = np.nan
    
    for i in range(n):
        if middle[i] > 1e-10 and not np.isnan(upper[i]) and not np.isnan(lower[i]):
            width[i] = (upper[i] - lower[i]) / middle[i]
    
    return width

def calculate_percentile_rank(values, window=100):
    """Percentile Rank of current value over rolling window"""
    n = len(values)
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(window, n):
        if not np.isnan(values[i]):
            window_vals = values[i-window+1:i+1]
            window_vals = window_vals[~np.isnan(window_vals)]
            if len(window_vals) > 0:
                pr[i] = np.sum(window_vals < values[i]) / len(window_vals) * 100.0
    
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
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)  # Fast RSI for 6h
    bb_upper, bb_middle, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    bb_width = calculate_bb_width(bb_upper, bb_lower, bb_middle)
    bb_width_pr = calculate_percentile_rank(bb_width, window=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_pr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (BOTH 1w AND 1d must agree) ===
        htf_bull = (close[i] > hma_1w_aligned[i]) and (close[i] > hma_1d_aligned[i])
        htf_bear = (close[i] < hma_1w_aligned[i]) and (close[i] < hma_1d_aligned[i])
        htf_neutral = not htf_bull and not htf_bear
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === RSI EXTREMES (Fast RSI(7): 25/75 for extremes, 35/65 for momentum) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 25.0
        rsi_extreme_overbought = rsi[i] > 75.0
        rsi_rising = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_falling = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        near_bb_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        above_bb_middle = close[i] > bb_middle[i]
        below_bb_middle = close[i] < bb_middle[i]
        
        # === BB WIDTH REGIME (squeeze = low vol, expansion = high vol) ===
        bb_squeeze = bb_width_pr[i] < 30.0  # Bottom 30% of BB width history
        bb_expansion = bb_width_pr[i] > 70.0  # Top 30% of BB width history
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # TREND LONG: HTF bull + (RSI recovery OR BB bounce OR HMA cross)
        if htf_bull:
            # Strong entry: RSI extreme oversold + near BB lower + HTF bull
            if rsi_extreme_oversold and near_bb_lower:
                desired_signal = SIZE_STRONG
            # Medium entry: RSI oversold + rising + above BB middle
            elif rsi_oversold and rsi_rising and above_bb_middle:
                desired_signal = SIZE_BASE
            # Momentum entry: RSI crossing above 50 + HTF bull + BB expansion
            elif rsi[i] > 50.0 and rsi[i-1] <= 50.0 and bb_expansion:
                desired_signal = SIZE_BASE
        
        # TREND SHORT: HTF bear + (RSI weakness OR BB rejection OR HMA cross)
        elif htf_bear:
            # Strong entry: RSI extreme overbought + near BB upper + HTF bear
            if rsi_extreme_overbought and near_bb_upper:
                desired_signal = -SIZE_STRONG
            # Medium entry: RSI overbought + falling + below BB middle
            elif rsi_overbought and rsi_falling and below_bb_middle:
                desired_signal = -SIZE_BASE
            # Momentum entry: RSI crossing below 50 + HTF bear + BB expansion
            elif rsi[i] < 50.0 and rsi[i-1] >= 50.0 and bb_expansion:
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION: Works when HTF neutral (range market)
        if desired_signal == 0.0 and htf_neutral:
            # Long: RSI extreme + near BB lower
            if rsi_extreme_oversold and near_bb_lower:
                desired_signal = SIZE_BASE
            # Short: RSI extreme + near BB upper
            elif rsi_extreme_overbought and near_bb_upper:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest since entry for trailing
            highest_since_entry = max(highest_since_entry, high[i])
            # Check stoploss
            if low[i] < stop_price:
                stoploss_triggered = True
            # Trail stop: move up as price rises
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            # Update lowest since entry for trailing
            lowest_since_entry = min(lowest_since_entry, low[i])
            # Check stoploss
            if high[i] > stop_price:
                stoploss_triggered = True
            # Trail stop: move down as price falls
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals