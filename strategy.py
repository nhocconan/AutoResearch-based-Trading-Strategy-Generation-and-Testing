#!/usr/bin/env python3
"""
Experiment #1586: 12h Primary + 1d HTF — Dual HMA Trend with RSI Confirmation

Hypothesis: After analyzing 1178 failed strategies, the key insight is:
1. 12h timeframe targets 20-50 trades/year - optimal fee efficiency
2. 1d HMA(21) provides proven trend bias filter (from best strategies)
3. Dual HMA crossover (8/21) on 12h gives entry timing with less lag than EMA
4. RSI(14) 40-60 filter ensures we enter on pullbacks, not extremes
5. ATR(14) 2.5x trailing stop controls drawdown
6. Looser entry conditions than failed experiments to ensure >10 trades/symbol

Why this should beat Sharpe 0.618:
- 12h TF has less noise than 4h, fewer whipsaws
- 1d HTF filter proven in mtf_1d_donchian_hma_rsi_1w_atr_v1 (current best)
- HMA reduces lag vs EMA while maintaining smoothness
- RSI 40-60 is looser than 45-55 (failed experiment #1571) = more trades
- Simple logic = more reliable signals across BTC/ETH/SOL

Timeframe: 12h (required for this experiment)
HTF: 1d HMA for trend bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
Position Size: 0.30 (discrete), Leverage: 1.0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_hma_1d_trend_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # Dual HMA for crossover signals
    hma_fast = calculate_hma(close, period=8)
    hma_slow = calculate_hma(close, period=21)
    
    # SMA 200 for long-term trend filter
    sma_200 = calculate_sma(close, period=200)
    
    # Donchian for breakout confirmation
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(sma_200[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === LONG TERM TREND (SMA 200) ===
        long_bull = close[i] > sma_200[i]
        long_bear = close[i] < sma_200[i]
        
        # === HMA CROSSOVER SIGNAL ===
        hma_cross_bull = hma_fast[i] > hma_slow[i]
        hma_cross_bear = hma_fast[i] < hma_slow[i]
        
        # Check previous bar for crossover detection
        hma_cross_bull_prev = False
        hma_cross_bear_prev = False
        if i > 0 and not np.isnan(hma_fast[i-1]) and not np.isnan(hma_slow[i-1]):
            hma_cross_bull_prev = hma_fast[i-1] > hma_slow[i-1]
            hma_cross_bear_prev = hma_fast[i-1] < hma_slow[i-1]
        
        # Fresh crossover (more reliable than sustained position)
        fresh_bull_cross = hma_cross_bull and not hma_cross_bull_prev
        fresh_bear_cross = hma_cross_bear and not hma_cross_bear_prev
        
        # === RSI FILTER (40-60 for pullback entries, looser than 45-55) ===
        rsi_bull = rsi[i] >= 40.0
        rsi_bear = rsi[i] <= 60.0
        
        # === BREAKOUT CONFIRMATION ===
        breakout_bull = close[i] > donch_upper[i] * 0.995  # Near upper band
        breakout_bear = close[i] < donch_lower[i] * 1.005  # Near lower band
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: Daily bull + Long bull + HMA bull + RSI support
        # Allow both fresh crossover AND sustained trend
        if daily_bull and long_bull and hma_cross_bull and rsi_bull:
            # Stronger signal on fresh crossover or breakout
            if fresh_bull_cross or breakout_bull:
                desired_signal = BASE_SIZE
            elif hma_fast[i] > hma_slow[i] * 1.002:  # Sustained with separation
                desired_signal = BASE_SIZE * 0.7
        
        # SHORT: Daily bear + Long bear + HMA bear + RSI support
        elif daily_bear and long_bear and hma_cross_bear and rsi_bear:
            # Stronger signal on fresh crossover or breakout
            if fresh_bear_cross or breakout_bear:
                desired_signal = -BASE_SIZE
            elif hma_fast[i] < hma_slow[i] * 0.998:  # Sustained with separation
                desired_signal = -BASE_SIZE * 0.7
        
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.7
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.7
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