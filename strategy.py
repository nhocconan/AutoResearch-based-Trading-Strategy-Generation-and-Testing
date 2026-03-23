#!/usr/bin/env python3
"""
Experiment #679: 4h Primary + 1d HTF — HMA Trend + RSI Pullback + Donchian Breakout

Hypothesis: Simplified 4h strategy with proven components beats complex regime switching.
Key learnings from 449 failed strategies:
1. Complex regime logic = 0 trades (too many filters never align)
2. RSI thresholds 20/80 = too strict, use 30/70 for trade generation
3. Donchian(20) works better than Donchian(50) for crypto volatility
4. HMA trend filter is proven edge (current best uses HMA)
5. MUST generate 30+ trades in train period or auto-reject

Strategy components:
- 1d HMA for macro trend bias (HTF filter)
- 4h HMA(21) for primary trend direction
- RSI(14) pullback entries at 35/65 (not extreme 20/80)
- Donchian(20) breakout confirmation
- ATR(14) trailing stop at 2.5x
- Position size: 0.25-0.30 discrete levels

Why this should beat Sharpe=0.612:
- Simpler = more trades = better statistics
- Proven HMA + RSI combination from current best strategy
- LOOSE thresholds ensure trade generation on ALL symbols
- 4h TF targets 20-50 trades/year (optimal fee/signal ratio)

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_simple_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_raw = 100 - (100 / (1 + rs))
        rsi[period:] = rsi_raw[period - 1:]
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channels — breakout detection."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    
    if n < period:
        return upper, lower, mid
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        mid[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donch_upper_4h, donch_lower_4h, donch_mid_4h = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF (1d) indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1d_raw = calculate_rsi(df_1d['close'].values, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size (28% of capital)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(50, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(hma_4h[i]) or np.isnan(rsi_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            continue
        if np.isnan(donch_upper_4h[i]) or np.isnan(donch_lower_4h[i]):
            continue
        
        # === HTF (1d) TREND BIAS ===
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        daily_rsi_bullish = rsi_1d_aligned[i] > 45
        daily_rsi_bearish = rsi_1d_aligned[i] < 55
        
        # === PRIMARY (4h) TREND ===
        hma_bullish = close[i] > hma_4h[i]
        hma_bearish = close[i] < hma_4h[i]
        hma_slope_up = hma_4h[i] > hma_4h[i - 3] if i >= 3 else False
        hma_slope_down = hma_4h[i] < hma_4h[i - 3] if i >= 3 else False
        
        # === RSI PULLBACK (LOOSE thresholds for trade generation) ===
        rsi_pullback_long = rsi_4h[i] < 55  # Pullback in uptrend
        rsi_pullback_short = rsi_4h[i] > 45  # Rally in downtrend
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donch_upper_4h[i - 1] if i > 0 else False
        breakout_short = close[i] < donch_lower_4h[i - 1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (any of these triggers) ===
        long_condition_1 = (
            daily_bullish and  # HTF bias
            hma_bullish and  # Primary trend
            rsi_pullback_long and  # Pullback entry
            rsi_4h[i] > 35  # Not crashing
        )
        
        long_condition_2 = (
            daily_bullish and
            breakout_long and  # Donchian breakout
            rsi_4h[i] < 65  # Not overbought
        )
        
        long_condition_3 = (
            hma_bullish and hma_slope_up and
            rsi_oversold and  # Deep pullback
            daily_rsi_bullish
        )
        
        if long_condition_1 or long_condition_2 or long_condition_3:
            desired_signal = SIZE
        
        # === SHORT ENTRY CONDITIONS (any of these triggers) ===
        short_condition_1 = (
            daily_bearish and  # HTF bias
            hma_bearish and  # Primary trend
            rsi_pullback_short and  # Rally entry
            rsi_4h[i] < 65  # Not pumping
        )
        
        short_condition_2 = (
            daily_bearish and
            breakout_short and  # Donchian breakdown
            rsi_4h[i] > 35  # Not oversold
        )
        
        short_condition_3 = (
            hma_bearish and hma_slope_down and
            rsi_overbought and  # Deep rally
            daily_rsi_bearish
        )
        
        if short_condition_1 or short_condition_2 or short_condition_3:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish AND RSI not extremely overbought
                if hma_bullish and rsi_4h[i] < 70:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if HMA still bearish AND RSI not extremely oversold
                if hma_bearish and rsi_4h[i] > 30:
                    desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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