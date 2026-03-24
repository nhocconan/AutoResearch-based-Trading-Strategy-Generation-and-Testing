#!/usr/bin/env python3
"""
Experiment #503: 6h Primary + 1d HTF — Volatility Spike Mean Reversion

Hypothesis: Volatility spikes (ATR ratio > 2.0) followed by mean reversion 
captured via Bollinger Band extremes is underutilized on 6h timeframe.
Unlike prior 6h experiments that used trend-following, this focuses on 
volatility exhaustion reversals with HTF trend filter.

Strategy logic:
1. 1d HMA(21) = daily trend bias (only trade reversion WITH trend)
2. 6h ATR(7)/ATR(30) ratio > 2.0 = volatility spike detected
3. 6h Bollinger(20, 2.5) extreme = price at statistical extreme
4. 6h RSI(14) confirmation = oversold/overbought extreme
5. Exit when ATR ratio < 1.3 (vol normalized) OR stoploss hit
6. Asymmetric sizing: stronger signals get 0.35, standard get 0.25

Why this might work on 6h:
- 6h captures multi-day vol cycles better than 4h
- Vol spikes often mark capitulation/exhaustion points
- HTF trend filter avoids catching falling knives
- Fewer trades than lower TF = less fee drag
- Mean reversion works well in 2022-2024 choppy periods

Target: Sharpe>0.40, trades>=120 train, trades>=20 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_volspike_bb_rsi_1d_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.5):
    """Bollinger Bands with configurable std deviation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi = calculate_rsi(close, period=14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger(close, period=20, std_mult=2.5)
    hma_21 = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Calculate ATR ratio (volatility spike detector)
    atr_ratio = np.zeros(n)
    atr_ratio[:] = np.nan
    for i in range(30, n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]):
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Position tracking for stoploss and exit logic
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_ratio[i]) or np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 2.0  # 2x normal volatility
        vol_normalizing = atr_ratio[i] < 1.3  # volatility returning to normal
        
        # === 1d HTF BIAS ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h TREND ===
        trend_bull = close[i] > hma_21[i]
        trend_bear = close[i] < hma_21[i]
        
        # === BOLLINGER BAND POSITION ===
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        bb_pct = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 1e-10 else 0.5
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        rsi_extreme_oversold = rsi[i] < 30.0
        rsi_extreme_overbought = rsi[i] > 70.0
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i] if not np.isnan(sma_50[i]) else True
        below_sma50 = close[i] < sma_50[i] if not np.isnan(sma_50[i]) else True
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Vol spike + BB lower + RSI oversold + HTF bull trend
        if vol_spike and at_bb_lower and rsi_oversold:
            if htf_bull and trend_bull:
                # Strong signal: all conditions align
                if rsi_extreme_oversold and bb_pct < 0.1:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            elif htf_bull:
                # HTF bull but 6h not confirmed - weaker signal
                desired_signal = SIZE_BASE * 0.8
            elif above_sma200:
                # Above long-term MA, mean reversion play
                desired_signal = SIZE_BASE * 0.7
        
        # SHORT: Vol spike + BB upper + RSI overbought + HTF bear trend
        elif vol_spike and at_bb_upper and rsi_overbought:
            if htf_bear and trend_bear:
                # Strong signal: all conditions align
                if rsi_extreme_overbought and bb_pct > 0.9:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
            elif htf_bear:
                # HTF bear but 6h not confirmed - weaker signal
                desired_signal = -SIZE_BASE * 0.8
            elif below_sma200:
                # Below long-term MA, mean reversion play
                desired_signal = -SIZE_BASE * 0.7
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        # Exit if volatility normalized (mean reversion complete)
        if in_position and vol_normalizing:
            exit_signal = True
        
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
            exit_signal = True
        
        if exit_signal:
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_7[i] if not np.isnan(atr_7[i]) else atr_30[i]
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