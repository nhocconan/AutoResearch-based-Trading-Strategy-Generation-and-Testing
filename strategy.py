#!/usr/bin/env python3
"""
Experiment #144: 4h Primary + 12h HTF — KAMA Trend + RSI Pullback + BB Regime

Hypothesis: After 143 experiments, 4h timeframe shows best balance of trade frequency and signal quality.
- KAMA (Kaufman Adaptive) adapts to volatility better than HMA/EMA for BTC/ETH whipsaws
- 12h HMA provides trend bias without being too restrictive (1d is too slow for 2025 bear)
- RSI pullback entries catch retracements in established trends (not breakout chasing)
- Bollinger Band regime switches between trend-follow and mean-revert modes
- LOOSE entry conditions to ensure >=30 trades on train, >=3 on test for ALL symbols

Key design:
- Timeframe: 4h (30-60 trades/year target, proven best for crypto)
- HTF: 12h HMA(50) for major trend bias
- Trend: KAMA(10,2,30) - adapts smoothing based on market efficiency
- Entry: RSI pullback (30-50 long, 50-70 short) + HTF confirmation
- Regime: BB Width for squeeze/expansion detection
- Position size: 0.30 (30% capital, conservative for 4h swings)
- Stoploss: 2.5x ATR trailing stop

Target: Sharpe>0.351, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_bb_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA[i] = KAMA[i-1] + SC * (close[i] - KAMA[i-1])
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    signal = np.abs(close - np.roll(close, period))
    signal[:period] = 0.0
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = np.abs(close[i] - close[i-1])
    
    noise_sum = np.zeros(n)
    for i in range(period, n):
        noise_sum[i] = np.sum(noise[i-period+1:i+1])
    
    er = np.zeros(n)
    for i in range(period, n):
        if noise_sum[i] > 1e-10:
            er[i] = signal[i] / noise_sum[i]
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Bollinger Bands with width for regime detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    width = (upper - lower) / (sma + 1e-10)
    
    return upper, lower, width

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # Calculate BB Width rolling median for regime detection
    bb_width_median = pd.Series(bb_width).rolling(window=100, min_periods=50).median().values
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 4h)
    
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
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
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
        if np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === 4h KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === REGIME DETECTION (BB Width) ===
        # BB Width > median = expansion (trend follow)
        # BB Width < median = squeeze (mean revert)
        is_expansion = False
        is_squeeze = False
        if not np.isnan(bb_width_median[i]):
            is_expansion = bb_width[i] > bb_width_median[i]
            is_squeeze = bb_width[i] <= bb_width_median[i]
        
        # === BB POSITION (for mean reversion) ===
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 1e-10:
            bb_position = (close[i] - bb_lower[i]) / bb_range
        else:
            bb_position = 0.5
        near_bb_lower = bb_position < 0.20
        near_bb_upper = bb_position > 0.80
        
        # === DESIRED SIGNAL (Loose conditions for trade generation) ===
        desired_signal = 0.0
        
        if is_expansion:
            # TREND REGIME: Follow trend with RSI pullback (LOOSE)
            # LONG: HTF bull + KAMA bull + RSI < 55 (pullback, not extreme)
            if htf_bull and kama_bull and rsi[i] < 55.0:
                desired_signal = SIZE
            # SHORT: HTF bear + KAMA bear + RSI > 45 (pullback, not extreme)
            elif htf_bear and kama_bear and rsi[i] > 45.0:
                desired_signal = -SIZE
        else:
            # SQUEEZE REGIME: Mean revert at BB bounds
            # LONG: near BB lower + RSI oversold
            if near_bb_lower and rsi[i] < 40.0:
                desired_signal = SIZE
            # SHORT: near BB upper + RSI overbought
            elif near_bb_upper and rsi[i] > 60.0:
                desired_signal = -SIZE
            # Fallback: KAMA trend with loose RSI
            elif kama_bull and rsi[i] < 50.0:
                desired_signal = SIZE * 0.7
            elif kama_bear and rsi[i] > 50.0:
                desired_signal = -SIZE * 0.7
        
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