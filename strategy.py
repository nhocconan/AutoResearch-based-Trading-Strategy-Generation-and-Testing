#!/usr/bin/env python3
"""
Experiment #1196: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Donchian

Hypothesis: Previous 12h attempts failed due to overly complex regime logic causing 0 trades.
This version uses PROVEN pattern from history: HMA crossover + RSI filter + ATR trail (SOL +0.879).

Key design:
- 1d HMA(50) for macro trend direction (from mtf_data helper)
- 12h HMA(21) crossover for entry timing
- RSI(14) pullback confirmation (not extreme, just direction-aligned)
- Donchian(20) breakout as additional confirmation
- ATR(14) 2.5x trailing stoploss
- Simple signal flip logic (no complex position tracking state)

Why this should work:
1. Fewer confluence requirements = more trades (target 30-50/year)
2. HMA reduces lag vs EMA (proven in best strategies)
3. RSI filter prevents chasing breakouts
4. 12h TF naturally limits trade frequency (low fee drag)
5. Discrete signal sizes (0, ±0.30) minimize churn

Position Size: 0.30 (discrete)
Stoploss: 2.5x ATR trailing
Target: Sharpe > 0.612, Trades > 30/train, > 3/test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_trend_rsi_pullback_donchian_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout indicator."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(hma_12h[i]) or np.isnan(hma_12h_fast[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(donchian_upper[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA50) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA crossover) ===
        hma_bull = hma_12h_fast[i] > hma_12h[i]
        hma_bear = hma_12h_fast[i] < hma_12h[i]
        
        # === RSI FILTER (pullback in direction of trend) ===
        rsi_bull = rsi[i] > 45.0 and rsi[i] < 70.0  # bullish but not overbought
        rsi_bear = rsi[i] < 55.0 and rsi[i] > 30.0  # bearish but not oversold
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        # === ENTRY CONDITIONS (simplified for more trades) ===
        desired_signal = 0.0
        
        # Long entry: macro bull + HMA bull + (RSI bull OR breakout)
        if macro_bull and hma_bull:
            if rsi_bull or breakout_long:
                desired_signal = BASE_SIZE
        
        # Short entry: macro bear + HMA bear + (RSI bear OR breakout)
        elif macro_bear and hma_bear:
            if rsi_bear or breakout_short:
                desired_signal = -BASE_SIZE
        
        # Apply stoploss
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            new_side = int(np.sign(desired_signal))
            if position_side != new_side:
                # New position or flip
                position_side = new_side
                entry_price = close[i]
                entry_atr = atr[i]
                if position_side > 0:
                    highest_since_entry = close[i]
                    lowest_since_entry = float('inf')
                else:
                    lowest_since_entry = close[i]
                    highest_since_entry = 0.0
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            # Close position
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals