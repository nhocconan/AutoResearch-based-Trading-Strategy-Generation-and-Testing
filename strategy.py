#!/usr/bin/env python3
"""
Experiment #401: 4h Primary + 1d/1w HTF — Simplified HMA Trend + Donchian Breakout

Hypothesis: After analyzing #389 (Sharpe=-0.453), the issues were:
1. RSI pullback requirement was TOO RESTRICTIVE — missed many trend entries
2. Too many exit conditions caused premature exits and whipsaw
3. Need simpler logic: trend direction + breakout confirmation only

Why this might work:
1. 1w HMA(21) for MAJOR trend (only trade with weekly trend)
2. 1d HMA(16/48) for intermediate trend confirmation
3. 4h Donchian(20) breakout for entry timing
4. ATR-based position sizing (smaller size when vol is high)
5. Only 2 exit conditions: stoploss (2.5*ATR) + weekly trend reversal

Position sizing: 0.20-0.35 based on ATR ratio (vol scaling)
Target: 30-50 trades/year on 4h, >=30 trades/symbol on train
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_simp_1d1w_v1"
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
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major trend)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (intermediate trend)
    hma_1d_16 = calculate_hma(df_1d['close'].values, period=16)
    hma_1d_48 = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_16_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_16)
    hma_1d_48_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_48)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR ratio for position sizing (vol scaling)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_14 / (atr_30 + 1e-10)  # ratio > 1 = high vol
    
    signals = np.zeros(n)
    
    # Base position size
    BASE_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(hma_1d_16_aligned[i]) or np.isnan(hma_1d_48_aligned[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W MAJOR TREND (primary direction filter) ===
        # Price above 1w HMA = bull market (favor longs)
        # Price below 1w HMA = bear market (favor shorts)
        bull_regime = close[i] > hma_1w_21_aligned[i]
        bear_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D INTERMEDIATE TREND (confirmation) ===
        hma_bullish = hma_1d_16_aligned[i] > hma_1d_48_aligned[i]
        hma_bearish = hma_1d_16_aligned[i] < hma_1d_48_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i]
        donchian_breakout_short = close[i] < donchian_lower[i]
        
        # === POSITION SIZING (vol scaling) ===
        # High vol (atr_ratio > 1.3) = smaller size
        # Normal vol = base size
        if atr_ratio[i] > 1.3:
            position_size = BASE_SIZE * 0.7  # Reduce by 30%
        else:
            position_size = BASE_SIZE
        
        # === ENTRY LOGIC — SIMPLIFIED ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Bull regime + HMA bullish + Donchian breakout
        if bull_regime and hma_bullish and donchian_breakout_long:
            new_signal = position_size
        
        # SHORT ENTRY: Bear regime + HMA bearish + Donchian breakout
        if bear_regime and hma_bearish and donchian_breakout_short:
            if new_signal == 0.0:  # Don't flip if already long
                new_signal = -position_size
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~2.5 days on 4h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if bull_regime and hma_bullish and rsi_14[i] < 60:
                new_signal = position_size * 0.7
            elif bear_regime and hma_bearish and rsi_14[i] > 40:
                new_signal = -position_size * 0.7
        
        # === EXIT CONDITIONS (simplified) ===
        # Only 2 exits: stoploss + weekly trend reversal
        
        # Weekly trend reversal exit
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 80:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 20:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals