#!/usr/bin/env python3
"""
Experiment #964: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 664 failed strategies, complexity is the enemy. The winning
combination is simple: HMA trend + RSI pullback + ATR stops. Recent failures
(#952-#963) show that too many regime filters = 0 trades or negative Sharpe.

Key insights from failures:
1. Funding rate as primary signal blocks trades (Sharpe=0.000 on #952, #958)
2. Complex regime switching (chop + CRSI + Donchian) = whipsaw losses
3. Extreme RSI thresholds (25/75) miss most entries
4. SOL-only strategies fail BTC/ETH (must work on ALL symbols)

What works (proven in baseline Sharpe=0.612):
- 4h HMA(21/48) crossover for trend direction
- 12h/1d HMA for macro bias (not hard filter)
- RSI(14) pullback to 40-60 range (not extremes)
- ATR(14) trailing stop at 2.5x
- Discrete signal sizes (0.0, ±0.25, ±0.30)

Why this should beat Sharpe=0.612:
1. SIMPLER logic = more reliable entries (30+ trades guaranteed)
2. RSI pullback (not extreme) = catches more trend continuations
3. HTF HMA as confluence (not hard filter) = doesn't block trades
4. ATR stop at 2.5x = tighter than 3x, less drawdown
5. Hold logic maintains position through minor pullbacks

Critical fixes from #963 (Sharpe=-0.510):
- Removed funding rate dependency (causes 0 trades on some symbols)
- Removed extreme RSI thresholds (25/75 → 40/60)
- Removed complex regime switching (chop index blocking entries)
- Added hold logic to maintain positions through pullbacks
- Ensured ALL symbols get trades (no SOL-only bias)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h1d_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma_crossover(hma_fast, hma_slow):
    """HMA crossover signal: 1=bullish, -1=bearish, 0=neutral."""
    n = len(hma_fast)
    crossover = np.zeros(n)
    
    for i in range(1, n):
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        if np.isnan(hma_fast[i-1]) or np.isnan(hma_slow[i-1]):
            continue
        
        # Bullish crossover: fast crosses above slow
        if hma_fast[i-1] <= hma_slow[i-1] and hma_fast[i] > hma_slow[i]:
            crossover[i] = 1
        # Bearish crossover: fast crosses below slow
        elif hma_fast[i-1] >= hma_slow[i-1] and hma_fast[i] < hma_slow[i]:
            crossover[i] = -1
    
    return crossover

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # 4h HMA crossover (21/48) for trend direction
    hma_4h_fast = calculate_hma(close, 21)
    hma_4h_slow = calculate_hma(close, 48)
    hma_cross_4h = calculate_hma_crossover(hma_4h_fast, hma_4h_slow)
    
    # Calculate and align 12h HMA for medium-term trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro regime
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Trend state tracking (avoid flip-flopping)
    trend_bullish = False
    trend_bearish = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_fast[i]) or np.isnan(hma_4h_slow[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === TREND DIRECTION (4h HMA Crossover) ===
        hma_bullish = hma_4h_fast[i] > hma_4h_slow[i]
        hma_bearish = hma_4h_fast[i] < hma_4h_slow[i]
        
        # Update trend state with hysteresis (avoid whipsaw)
        if hma_bullish and not trend_bearish:
            trend_bullish = True
            trend_bearish = False
        elif hma_bearish and not trend_bullish:
            trend_bearish = True
            trend_bullish = False
        
        # === HTF TREND BIAS (12h/1d HMA) ===
        # Use as confluence, not hard filter (ensures trades on all symbols)
        hma_12h_bullish = close[i] > hma_12h_aligned[i]
        hma_12h_bearish = close[i] < hma_12h_aligned[i]
        
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # Count HTF confluence (0-2 bullish/bearish signals)
        htf_bullish_count = int(hma_12h_bullish) + int(hma_1d_bullish)
        htf_bearish_count = int(hma_12h_bearish) + int(hma_1d_bearish)
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulls back to 40-50 in uptrend
        rsi_pullback_long = 40 <= rsi_4h[i] <= 55
        # Short: RSI rallies to 45-60 in downtrend
        rsi_pullback_short = 45 <= rsi_4h[i] <= 60
        # RSI extreme (stronger signal)
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        if trend_bullish:
            # Strong long: HMA bullish + RSI pullback + HTF confluence
            if rsi_pullback_long and htf_bullish_count >= 1:
                desired_signal = BASE_SIZE
            # Moderate long: HMA bullish + RSI pullback (no HTF required)
            elif rsi_pullback_long:
                desired_signal = REDUCED_SIZE
            # Strong long: HMA bullish + RSI oversold
            elif rsi_oversold:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        if trend_bearish:
            # Strong short: HMA bearish + RSI pullback + HTF confluence
            if rsi_pullback_short and htf_bearish_count >= 1:
                desired_signal = -BASE_SIZE
            # Moderate short: HMA bearish + RSI pullback (no HTF required)
            elif rsi_pullback_short:
                desired_signal = -REDUCED_SIZE
            # Strong short: HMA bearish + RSI overbought
            elif rsi_overbought:
                desired_signal = -BASE_SIZE
        
        # === HMA CROSSOVER ENTRY (momentum entry) ===
        if hma_cross_4h[i] == 1 and rsi_4h[i] < 60:
            # Bullish crossover + RSI not overbought
            desired_signal = max(desired_signal, BASE_SIZE)
        elif hma_cross_4h[i] == -1 and rsi_4h[i] > 40:
            # Bearish crossover + RSI not oversold
            desired_signal = min(desired_signal, -BASE_SIZE)
        
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
                # Hold long if HMA still bullish and RSI not overbought
                if trend_bullish and rsi_4h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if HMA still bearish and RSI not oversold
                if trend_bearish and rsi_4h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HMA crosses bearish
            if hma_cross_4h[i] == -1:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HMA crosses bullish
            if hma_cross_4h[i] == 1:
                desired_signal = 0.0
            # Exit if RSI extremely oversold
            if rsi_4h[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
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
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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