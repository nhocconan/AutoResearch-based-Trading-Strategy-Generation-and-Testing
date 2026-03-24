#!/usr/bin/env python3
"""
Experiment #023: 1d Primary + 1w HTF — Donchian Breakout with Weekly Trend Filter

Hypothesis: After 19 failed experiments, the pattern is clear:
1. Over-filtering kills trade generation (many strategies got Sharpe=0.000)
2. Connors RSI + Choppiness failed repeatedly on 1d (#013, #017)
3. 1d timeframe needs SIMPLE logic with LOOSE thresholds to generate trades

This strategy uses:
1. 1w HMA(21) for simple trend bias (not over-filtered)
2. Donchian(20) breakout for entry triggers (proven on daily TF)
3. RSI(14) with LOOSE thresholds (30/70) to ensure trades generate
4. 3x ATR trailing stop for risk management

Key differences from failed 1d strategies:
- FEWER filters (no Choppiness, no Connors RSI complexity)
- LOOSER RSI thresholds (30/70 vs 25/75 or tighter)
- Donchian breakout is more reliable than EMA crossover on 1d
- 1w HTF used ONLY for trend direction, not entry timing

Entry Logic:
- Long: 1d close > Donchian_high(20) + 1w HMA bullish + RSI > 30
- Short: 1d close < Donchian_low(20) + 1w HMA bearish + RSI < 70
- Size: 0.30 (discrete, minimizes fee churn)

Target: Sharpe>0.25, trades>30/symbol train, >3/symbol test, DD>-40%
Timeframe: 1d (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_1w_loose_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    wma_diff = 2.0 * wma_half - wma_full
    hma = pd.Series(wma_diff).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    
    return hma

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        donchian_high[i] = np.max(high[i - period + 1:i + 1])
        donchian_low[i] = np.min(low[i - period + 1:i + 1])
    
    return donchian_high, donchian_low

def calculate_rsi(close, period=14):
    """RSI - momentum filter with loose thresholds"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for HTF trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Discrete position size (max 0.40 per rules)
    
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_high[i]):
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
        
        # === 1W HMA TREND BIAS (simple, not over-filtered) ===
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        breakout_long = close[i] > donchian_high[i]
        breakout_short = close[i] < donchian_low[i]
        
        # === RSI FILTER (LOOSE thresholds to ensure trades) ===
        rsi_ok_long = rsi[i] > 30.0  # Not extremely oversold
        rsi_ok_short = rsi[i] < 70.0  # Not extremely overbought
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Donchian breakout + 1w HMA bullish + RSI filter
        if breakout_long and hma_1w_bull and rsi_ok_long:
            desired_signal = SIZE
        
        # Short entry: Donchian breakout + 1w HMA bearish + RSI filter
        elif breakout_short and hma_1w_bear and rsi_ok_short:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
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