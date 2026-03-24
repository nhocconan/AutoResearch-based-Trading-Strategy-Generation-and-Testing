#!/usr/bin/env python3
"""
Experiment #1614: 4h Primary + 12h/1d HTF — Simplified Regime + RSI Pullback + Donchian Breakout

Hypothesis: #1604 had Sharpe=0.017 but low return. The regime logic was too complex and
restrictive. This version SIMPLIFIES: use 12h/1d HMA for trend bias ONLY, RSI(14) for
pullback entries (simpler than CRSI, more trades), Donchian(20) for breakout confirmation.

Key changes from #1604:
1. REMOVED Choppiness Index - too many false regime switches, adds complexity
2. RSI(14) instead of CRSI - simpler, generates more trades (RSI<35 long, >65 short)
3. Donchian(20) breakout - adds momentum confirmation (proven in SOL experiments)
4. LOOSER thresholds - RSI 35/65 instead of CRSI 20/80, ensures trade generation
5. Tighter stoploss - 2.0x ATR instead of 2.5x (more exits = more re-entry opportunities)
6. Simpler position tracking - reset on signal flip, not just stoploss
7. BASE_SIZE = 0.30 (discrete: 0.0, ±0.30)

Why this should beat Sharpe 0.618:
- Simpler logic = fewer edge cases = more consistent signals
- RSI pullback in trend = high win rate entries (buy dips in uptrend)
- Donchian breakout = captures momentum moves
- 12h + 1d dual HTF = strong trend filter without overfitting
- LOOSE entry conditions = guarantees >10 trades/symbol train, >3 test

Timeframe: 4h (required)
HTF: 12h HMA + 1d HMA for trend bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 30/symbol train, > 5/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_pullback_donchian_12h1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """RSI with proper min_periods"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(close)
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for intermediate trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for long-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking
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
            in_position = False
            position_side = 0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND BIAS (12h + 1d HMA) ===
        # Both HTF must agree for strong trend signal
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        intermediate_bull = close[i] > hma_12h_aligned[i]
        intermediate_bear = close[i] < hma_12h_aligned[i]
        
        # Strong bull: both 12h and 1d bullish
        strong_bull = daily_bull and intermediate_bull
        # Strong bear: both 12h and 1d bearish
        strong_bear = daily_bear and intermediate_bear
        # Neutral: mixed signals
        neutral = not strong_bull and not strong_bear
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI oversold (<35) in bullish trend
        rsi_oversold = rsi[i] < 35.0
        # Short: RSI overbought (>65) in bearish trend
        rsi_overbought = rsi[i] > 65.0
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        # Long breakout: price near Donchian upper (momentum)
        donch_breakout_long = close[i] > donch_upper[i] * 0.995
        # Short breakout: price near Donchian lower (momentum)
        donch_breakout_short = close[i] < donch_lower[i] * 1.005
        
        # === PRIMARY SIGNAL LOGIC ===
        desired_signal = 0.0
        
        # MODE 1: TREND FOLLOWING with RSI pullback (highest probability)
        # Long: Strong bull trend + RSI pullback (buy the dip)
        if strong_bull and rsi_oversold:
            desired_signal = BASE_SIZE
        # Short: Strong bear trend + RSI overbought (sell the rip)
        elif strong_bear and rsi_overbought:
            desired_signal = -BASE_SIZE
        
        # MODE 2: DONCHIAN BREAKOUT (momentum continuation)
        # Only take breakouts in direction of 1d trend
        elif donch_breakout_long and daily_bull:
            desired_signal = BASE_SIZE
        elif donch_breakout_short and daily_bear:
            desired_signal = -BASE_SIZE
        
        # MODE 3: NEUTRAL REGIME - mean reversion at extremes
        # Only if RSI is VERY extreme (<20 or >80)
        elif neutral:
            if rsi[i] < 20.0:
                desired_signal = BASE_SIZE * 0.5  # Half size in neutral
            elif rsi[i] > 80.0:
                desired_signal = -BASE_SIZE * 0.5  # Half size in neutral
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.4:
            final_signal = -BASE_SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                # New position
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