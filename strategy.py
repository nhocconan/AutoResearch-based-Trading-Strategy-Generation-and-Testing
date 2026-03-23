#!/usr/bin/env python3
"""
Experiment #1113: 1d Primary + 1w HTF — Simplified Multi-Timeframe Trend Pullback

Hypothesis: After 800+ failed experiments, the pattern is clear:
1. OVER-FILTERING is the #1 cause of 0 trades (ADX, Choppiness, CRSI together = no signals)
2. 1d timeframe naturally produces 20-50 trades/year — perfect for fee efficiency
3. 1w HMA provides ULTRA-macro trend filter with minimal whipsaws
4. LOOSE RSI thresholds (45/55) ensure adequate trade frequency
5. Simple ATR trailing stop (2.5x) protects capital without premature exits
6. DISCRETE position sizing (0.0, ±0.30) minimizes fee churn

Why this should beat Sharpe=0.612:
- 1w trend filter is SLOWER and more reliable than 1d/4h
- Fewer filters = MORE trades (critical for Sharpe calculation)
- 1d has less noise than 4h/12h for entry timing
- Proven in research: HMA + RSI + ATR worked on SOL (Sharpe +0.879)
- Position size 0.30 balances return vs 2022 crash risk (77% → 23% equity loss)

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete)
Stoploss: 2.5x ATR trailing
Target: 25-40 trades/year per symbol, Sharpe > 0.612, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_1w_atr_simple_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    sqrt_period = max(1, int(np.sqrt(period)))
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator (0-100)."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
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

def calculate_sma(close, period=50):
    """Simple Moving Average for additional trend filter."""
    n = len(close)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for ultra-macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    rsi_1d = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, period=50)
    
    # 1d HMA for intermediate trend
    hma_1d = calculate_hma(close, period=21)
    
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
        if np.isnan(rsi_1d[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d[i]):
            continue
        if np.isnan(sma_50[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === ULTRA-MACRO TREND (1w HMA) ===
        # This is the PRIMARY filter — only trade in direction of weekly trend
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA + SMA50) ===
        # Confirms 1w signal with 1d trend
        intermediate_bull = close[i] > hma_1d[i] and close[i] > sma_50[i]
        intermediate_bear = close[i] < hma_1d[i] and close[i] < sma_50[i]
        
        # === PULLBACK SIGNAL (1d RSI) ===
        # LOOSE thresholds to ensure trade frequency
        # Long: RSI < 55 (pullback in uptrend)
        # Short: RSI > 45 (rally in downtrend)
        rsi_pullback_long = rsi_1d[i] < 55.0
        rsi_pullback_short = rsi_1d[i] > 45.0
        
        # === ENTRY CONDITIONS (SIMPLE — 2 filters only) ===
        desired_signal = 0.0
        
        # LONG: 1w bull + 1d bull + RSI pullback
        if macro_bull and intermediate_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE
        
        # SHORT: 1w bear + 1d bear + RSI pullback
        elif macro_bear and intermediate_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        
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
                # Hold long if 1w still bull
                if macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1w still bear
                if macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1w trend reverses
            if macro_bear:
                desired_signal = 0.0
            # Exit long if RSI very overbought (>75)
            elif rsi_1d[i] > 75.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1w trend reverses
            if macro_bull:
                desired_signal = 0.0
            # Exit short if RSI very oversold (<25)
            elif rsi_1d[i] < 25.0:
                desired_signal = 0.0
        
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