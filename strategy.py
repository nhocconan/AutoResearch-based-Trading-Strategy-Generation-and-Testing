#!/usr/bin/env python3
"""
Experiment #526: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 471 failed strategies (mostly complex volspike/choppiness combos),
return to PROVEN simplicity on higher timeframes. 12h should balance trade frequency
(20-50/year) with signal quality.

Key insights from failures:
- Complex multi-condition entries = 0 trades or negative Sharpe
- Volatility spike strategies: ALL failed (15+ experiments)
- Choppiness Index: Failed 4+ times
- Lower TFs (15m-4h): Consistently negative Sharpe
- 12h/1d: Best historical performance (current best Sharpe=0.435)

This strategy uses SIMPLER logic than #521:
1. 1d HMA(21) for major trend direction (only trade with HTF trend)
2. 12h HMA(16/48) crossover for entry timing
3. RSI(14) with WIDER thresholds (30/70) to ensure trade frequency
4. ATR(14) 3x trailing stop for risk management
5. Single entry condition per direction (no conflicting filters)

Why this might work:
- 12h TF = less noise than 4h, more trades than 1d
- Simpler entry logic = consistent signals across BTC/ETH/SOL
- Wider RSI thresholds = more trades (address #1 failure mode)
- 1d trend filter = prevents counter-trend trades in bear markets
- Discrete position sizing (0.30) = minimal fee churn

Position sizing: 0.30 (discrete, max 0.40 per rules)
Stoploss: 3.0 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_rsi_simp_1d_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # HMA crossover signals (16/48)
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    
    # RSI for entry filter
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # === 12H HMA CROSSOVER SIGNALS ===
        # Bullish crossover: fast HMA crosses above slow HMA
        hma_cross_up = (hma_12h_16[i] > hma_12h_48[i]) and (hma_12h_16[i-1] <= hma_12h_48[i-1])
        # Bearish crossover: fast HMA crosses below slow HMA
        hma_cross_down = (hma_12h_16[i] < hma_12h_48[i]) and (hma_12h_16[i-1] >= hma_12h_48[i-1])
        
        # === RSI FILTER (WIDER thresholds for trade frequency) ===
        rsi_not_overbought = rsi_14[i] < 70.0
        rsi_not_oversold = rsi_14[i] > 30.0
        
        # === ENTRY LOGIC — SIMPLE, SINGLE CONDITION PER DIRECTION ===
        new_signal = 0.0
        
        # LONG: HMA crossover up + bull regime + RSI not overbought
        if hma_cross_up and bull_regime and rsi_not_overbought:
            new_signal = POSITION_SIZE
        
        # SHORT: HMA crossover down + bear regime + RSI not oversold
        elif hma_cross_down and bear_regime and rsi_not_oversold:
            new_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME FLIP ===
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals