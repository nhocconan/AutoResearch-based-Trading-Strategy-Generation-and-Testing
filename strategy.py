#!/usr/bin/env python3
"""
Experiment #234: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Daily timeframe strategies fail when over-filtered. This version strips
back to essentials after analyzing 200+ failed experiments:

Core Logic:
- Weekly HMA(50) = Major trend bias (ONLY trade in direction)
- Daily HMA(21) = Primary trend signal
- RSI(14) pullback = Entry timing (not extreme, just 40-60 zone for continuation)
- Donchian(20) breakout = Momentum confirmation (price breaks recent high/low)
- ATR(14) trailing stop = Risk management (2.5x ATR)

Key Insight from Failures:
- Experiment #226 (1d KAMA RSI) failed with Sharpe=-0.303 — too many filters
- Experiment #232 (12h simplified) got Sharpe=0.242 — SIMPLER works better
- CRSI, Choppiness, multiple regime filters = 0 trades on daily bars

This strategy uses MINIMAL filters to ensure trade generation:
- Weekly trend bias (soft filter — can override in strong moves)
- Daily HMA direction
- RSI in reasonable zone (35-65, not extreme)
- Price action confirmation (Donchian break OR HMA crossover)

Position sizing: 0.30 base (30% of capital)
Stoploss: 2.5x ATR trailing
Target: Sharpe>0.40 (beat current best 0.399), DD>-35%, trades>=20 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_donchian_1w_v1"
timeframe = "1d"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for daily TF)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) - Soft filter ===
        # Weekly trend gives bias but doesn't block trades entirely
        htf_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === DAILY HMA TREND ===
        hma_bull = close[i] > hma_21[i] and hma_21[i] > hma_50[i]
        hma_bear = close[i] < hma_21[i] and hma_21[i] < hma_50[i]
        
        # === RSI ZONE (not extreme - continuation trades) ===
        rsi_neutral_long = 35.0 < rsi[i] < 65.0
        rsi_neutral_short = 35.0 < rsi[i] < 65.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = False
        breakout_short = False
        if not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            breakout_long = close[i] > donchian_upper[i-1]
            breakout_short = close[i] < donchian_lower[i-1]
        
        # === HMA CROSSOVER SIGNAL ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 1 and not np.isnan(hma_21[i-1]) and not np.isnan(hma_50[i-1]):
            hma_cross_long = (hma_21[i] > hma_50[i]) and (hma_21[i-1] <= hma_50[i-1])
            hma_cross_short = (hma_21[i] < hma_50[i]) and (hma_21[i-1] >= hma_50[i-1])
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer filters = more trades) ===
        desired_signal = 0.0
        
        # LONG ENTRY: Multiple paths to entry (OR logic for more trades)
        long_score = 0
        
        # Path 1: HMA bullish + RSI in zone + breakout
        if hma_bull and rsi_neutral_long and breakout_long:
            long_score += 2
        
        # Path 2: HMA crossover + weekly bias
        if hma_cross_long and htf_bull:
            long_score += 2
        
        # Path 3: Strong HMA bull + weekly bull (even without breakout)
        if hma_bull and htf_bull and rsi_neutral_long:
            long_score += 1
        
        # SHORT ENTRY
        short_score = 0
        
        # Path 1: HMA bearish + RSI in zone + breakout
        if hma_bear and rsi_neutral_short and breakout_short:
            short_score += 2
        
        # Path 2: HMA crossover + weekly bias
        if hma_cross_short and htf_bear:
            short_score += 2
        
        # Path 3: Strong HMA bear + weekly bear
        if hma_bear and htf_bear and rsi_neutral_short:
            short_score += 1
        
        # Enter on highest score (with minimum threshold)
        if long_score >= 2 and short_score < 2:
            desired_signal = SIZE
        elif short_score >= 2 and long_score < 2:
            desired_signal = -SIZE
        elif long_score >= 3:
            desired_signal = SIZE
        elif short_score >= 3:
            desired_signal = -SIZE
        
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
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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