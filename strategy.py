#!/usr/bin/env python3
"""
Experiment #1164: 4h Primary + 12h HTF — KAMA Trend + Donchian Breakout + Light RSI

Hypothesis: After analyzing 850+ failed experiments, clear patterns emerge:
- 4h timeframe with 12h HTF filter works (see #1154 Sharpe=0.150 positive)
- KAMA adapts better to crypto volatility than HMA/EMA (Efficiency Ratio)
- Donchian(20) breakout catches major moves without whipsaw
- Light RSI filter (only avoid extremes) prevents bad entries without killing trades
- Minimal exit conditions let winners run (trailing stop only)
- 12h HTF prevents counter-trend trades in major moves

Why this should beat Sharpe=0.612:
- Simpler logic = fewer 0-trade failures (#1159, #1157, #1155 all had 0 trades)
- KAMA adapts to volatility regimes better than fixed HMA
- Donchian breakout + trend filter = proven combo (#1154 had positive Sharpe)
- Position size 0.30 discrete balances returns vs drawdown
- Target: 30-50 trades/year on 4h, Sharpe > 0.612

Timeframe: 4h (primary)
HTF: 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing (appropriate for 4h volatility)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_donchian_12h_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman's Adaptive Moving Average — adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER (trending) = less smoothing, Low ER (choppy) = more smoothing
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout detection.
    Upper = highest high over period, Lower = lowest low over period
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    RSI < 30 = oversold, RSI > 70 = overbought
    We use light filter: avoid RSI > 80 for long, RSI < 20 for short
    """
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h KAMA for macro trend filter
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    kama_4h = calculate_kama(close, period=10)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi_4h[i]) or np.isnan(kama_4h[i]):
            continue
        if np.isnan(kama_12h_aligned[i]) or np.isnan(donchian_upper[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (12h KAMA) ===
        macro_bull = close[i] > kama_12h_aligned[i]
        macro_bear = close[i] < kama_12h_aligned[i]
        
        # === LOCAL TREND (4h KAMA) ===
        local_bull = close[i] > kama_4h[i]
        local_bear = close[i] < kama_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        # Break above previous upper band for long, below lower for short
        breakout_long = close[i] > donchian_upper[i - 1]
        breakout_short = close[i] < donchian_lower[i - 1]
        
        # === RSI FILTER (avoid extremes only) ===
        # Only avoid RSI > 80 for long, RSI < 20 for short (light filter)
        rsi_ok_long = rsi_4h[i] < 80.0
        rsi_ok_short = rsi_4h[i] > 20.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + local bull + Donchian breakout + RSI not extreme
        if macro_bull and local_bull and breakout_long and rsi_ok_long:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + local bear + Donchian breakout + RSI not extreme
        elif macro_bear and local_bear and breakout_short and rsi_ok_short:
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and macro_bear:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and macro_bull:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro and local still bull
                if macro_bull and local_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro and local still bear
                if macro_bear and local_bear:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals