#!/usr/bin/env python3
"""
Experiment #479: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend Filter

Hypothesis: After 478 experiments, 4h strategies fail because:
1. Mean-reversion (CRSI/RSI) whipsaws in trending markets
2. Too many conflicting filters = 0 trades or negative Sharpe
3. Need cleaner breakout signals with strong HTF trend filter

Why Donchian breakout might beat Sharpe=0.435:
- Proven on SOL (Sharpe +0.782 in research notes)
- Clear breakout signals, less subjective than RSI extremes
- 1d HMA(21) filters out counter-trend trades (major whipsaw killer)
- Naturally produces 20-50 trades/year on 4h (optimal fee/edge ratio)
- Works in both bull and bear regimes (directional breakouts)

Strategy logic:
- 1d HMA(21) = major trend direction (ONLY trade with trend)
- 4h Donchian(20) breakout = entry trigger
- 4h RSI(14) momentum confirmation (>55 long, <45 short)
- ATR(14) 2.5x trailing stop (wider for 4h noise)
- Position size: 0.30 long, 0.25 short (asymmetric)

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_1d_breakout_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel.
    Upper = highest high over N periods
    Lower = lowest low over N periods
    Breakout = price crosses above upper or below lower
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    sma_200 = calculate_sma(close, period=200)
    
    # Track previous bar values for breakout detection
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    prev_donchian_upper = np.roll(donchian_upper, 1)
    prev_donchian_lower = np.roll(donchian_lower, 1)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(sma_200[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter - ONLY trade with trend) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # Stronger trend confirmation on 1d
        strong_bull = bull_regime and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        strong_bear = bear_regime and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H DONCHIAN BREAKOUT DETECTION ===
        # Long breakout: price crosses ABOVE Donchian upper
        long_breakout = (close[i] > donchian_upper[i]) and (prev_close[i] <= prev_donchian_upper[i])
        # Short breakout: price crosses BELOW Donchian lower
        short_breakout = (close[i] < donchian_lower[i]) and (prev_close[i] >= prev_donchian_lower[i])
        
        # === 4H RSI MOMENTUM CONFIRMATION ===
        rsi_bullish = rsi_14[i] > 50.0
        rsi_bearish = rsi_14[i] < 50.0
        rsi_strong_bull = rsi_14[i] > 55.0
        rsi_strong_bear = rsi_14[i] < 45.0
        
        # === SMA200 FILTER (additional trend confirmation) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — BREAKOUT WITH TREND FILTER ===
        new_signal = 0.0
        
        # LONG ENTRIES (breakout + trend alignment)
        if strong_bull and long_breakout and rsi_strong_bull:
            new_signal = LONG_SIZE
        elif bull_regime and long_breakout and rsi_bullish and above_sma200:
            new_signal = LONG_SIZE * 0.8
        elif strong_bull and rsi_strong_bull and close[i] > donchian_upper[i] * 0.99:
            # Near breakout with strong momentum
            new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (breakout + trend alignment)
        if new_signal == 0.0:
            if strong_bear and short_breakout and rsi_strong_bear:
                new_signal = -SHORT_SIZE
            elif bear_regime and short_breakout and rsi_bearish and below_sma200:
                new_signal = -SHORT_SIZE * 0.8
            elif strong_bear and rsi_strong_bear and close[i] < donchian_lower[i] * 1.01:
                # Near breakout with strong momentum
                new_signal = -SHORT_SIZE * 0.7
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long when RSI overbought or trend flips
        if in_position and position_side > 0:
            if rsi_14[i] > 75.0:
                new_signal = 0.0
            elif bear_regime and hma_1d_21_aligned[i] < hma_1d_50_aligned[i]:
                new_signal = 0.0
        
        # Exit short when RSI oversold or trend flips
        if in_position and position_side < 0:
            if rsi_14[i] < 25.0:
                new_signal = 0.0
            elif bull_regime and hma_1d_21_aligned[i] > hma_1d_50_aligned[i]:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals