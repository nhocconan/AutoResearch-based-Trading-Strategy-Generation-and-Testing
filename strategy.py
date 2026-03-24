#!/usr/bin/env python3
"""
Experiment #837: 15m Primary + 4h/12h HTF — Mean Reversion with HTF Bias

Hypothesis: 15m timeframe needs LOOSE entry conditions to generate trades.
Previous 15m experiments failed with 0 trades (Sharpe=0.000) due to overly
strict HTF filters. This version uses HTF as SOFT BIAS, not hard filter.

Key innovations:
1. 4h HMA(21) for trend BIAS only — prefer longs in bull, but allow counter-trend
2. 15m RSI(7) with loose thresholds (25/75 not 30/70) for more entries
3. 15m Bollinger Band squeeze for volatility confirmation
4. Session filter: 00-12 UTC only (London/NY overlap) — reduces trade count
5. Force entry after 20 bars of no signal on RSI extreme (guarantees trades)
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)
7. ATR(14) 2.0x trailing stop for tighter risk on lower TF

Target: Sharpe>0.40, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.15-0.20 discrete
Trade freq: 40-100/year (session filter + HTF bias)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_bb_session_4h12h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    """Bollinger Bands - volatility bands for mean reversion"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Band width as % of price
    
    return upper, lower, width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # BB width percentile for squeeze detection
    bb_width_sma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_std = pd.Series(bb_width).rolling(window=50, min_periods=50).std().values
    bb_width_zscore = (bb_width - bb_width_sma) / (bb_width_std + 1e-10)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    bars_since_signal = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER: 00-12 UTC only (London/NY overlap) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = (hour_utc >= 0) and (hour_utc < 12)
        
        # === HTF BIAS (4h/12h HMA) — SOFT BIAS, NOT HARD FILTER ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both agree
        htf_strong_bull = htf_4h_bull and htf_12h_bull
        htf_strong_bear = htf_4h_bear and htf_12h_bear
        
        # === RSI CONDITIONS (LOOSE for more trades on 15m) ===
        rsi_oversold = rsi_7[i] < 25.0
        rsi_overbought = rsi_7[i] > 75.0
        rsi_extreme_oversold = rsi_7[i] < 15.0
        rsi_extreme_overbought = rsi_7[i] > 85.0
        
        # === BOLLINGER CONDITIONS ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        bb_squeeze = bb_width_zscore[i] < -1.0  # Width below average
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: In session + (RSI oversold OR price below BB)
        # HTF bias increases size but doesn't block entry
        if in_session:
            if rsi_oversold or price_below_bb:
                if htf_strong_bull:
                    # Strong bull bias = larger size
                    if rsi_extreme_oversold or price_below_bb:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
                elif htf_4h_bull:
                    # Mild bull bias
                    if rsi_extreme_oversold:
                        desired_signal = SIZE_BASE
                    else:
                        desired_signal = SIZE_BASE * 0.75
                else:
                    # Counter-trend (bear HTF) = only on extreme
                    if rsi_extreme_oversold and bb_squeeze:
                        desired_signal = SIZE_BASE * 0.5
        
        # SHORT: In session + (RSI overbought OR price above BB)
            elif rsi_overbought or price_above_bb:
                if htf_strong_bear:
                    # Strong bear bias = larger size
                    if rsi_extreme_overbought or price_above_bb:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
                elif htf_4h_bear:
                    # Mild bear bias
                    if rsi_extreme_overbought:
                        desired_signal = -SIZE_BASE
                    else:
                        desired_signal = -SIZE_BASE * 0.75
                else:
                    # Counter-trend (bull HTF) = only on extreme
                    if rsi_extreme_overbought and bb_squeeze:
                        desired_signal = -SIZE_BASE * 0.5
        
        # === FORCE ENTRY AFTER 20 BARS OF NO SIGNAL ===
        # This guarantees trade generation (critical for 15m)
        bars_since_signal += 1
        if bars_since_signal >= 20 and desired_signal == 0.0:
            if rsi_extreme_oversold and in_session:
                desired_signal = SIZE_BASE * 0.5
            elif rsi_extreme_overbought and in_session:
                desired_signal = -SIZE_BASE * 0.5
        
        if desired_signal != 0.0:
            bars_since_signal = 0
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.4:
            # Small positions rounded to base size
            final_signal = np.sign(desired_signal) * SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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