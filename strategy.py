#!/usr/bin/env python3
"""
Experiment #851: 4h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 587+ failed strategies, complex regime switching is the problem.
This strategy uses SIMPLER logic with wider thresholds to ensure trade generation.

Key insight from failures:
- Complex regime filters (CHOP + Fisher + multiple conditions) = 0 trades
- Need wider RSI thresholds (25/75 not 30/70) for 4h timeframe
- 1d HMA21 for trend bias (simpler than SMA200)
- Hold logic to maintain positions through pullbacks
- ATR stoploss at 2.0x (tighter than 2.5x for 4h)

Strategy design:
1. 4h Primary timeframe (target 30-50 trades/year)
2. 1d HMA(21) for long-term trend bias
3. 4h HMA(21) for short-term trend
4. 4h RSI(14) with 25/75 thresholds (wider = more signals)
5. 4h ATR(14) for trailing stop (2.0x)
6. Simple entry: RSI extreme + aligned with 1d trend
7. Hold through minor pullbacks (RSI not crossed 50)
8. Discrete sizes: 0.0, ±0.25, ±0.30

Why this should work:
- Simpler = more trades (addresses #1 failure mode)
- 1d HMA smoother than SMA200 for trend filtering
- RSI 25/75 captures more reversals on 4h
- Hold logic prevents premature exits

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_simple_1d1w_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average — faster response than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    rsi_4h = calculate_rsi(close, period=14)
    hma_4h = calculate_hma(close, period=21)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d HMA for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss and hold logic
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # RSI thresholds (wider for 4h to ensure trades)
    RSI_LONG_ENTRY = 25
    RSI_SHORT_ENTRY = 75
    RSI_EXIT_LONG = 55
    RSI_EXIT_SHORT = 45
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]) or np.isnan(hma_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if atr_4h[i] <= 1e-10:
            continue
        
        # === LONG-TERM TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === SHORT-TERM TREND (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h[i]
        trend_4h_bearish = close[i] < hma_4h[i]
        
        # === RSI SIGNALS (Wider thresholds for 4h) ===
        rsi_oversold = rsi_4h[i] < RSI_LONG_ENTRY
        rsi_overbought = rsi_4h[i] > RSI_SHORT_ENTRY
        rsi_recovering = rsi_4h[i] > 30 and rsi_4h[i] < 50
        rsi_weakening = rsi_4h[i] < 70 and rsi_4h[i] > 50
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        # Primary: RSI oversold + aligned with 1d trend
        if rsi_oversold and trend_1d_bullish:
            desired_signal = BASE_SIZE
        # Secondary: RSI recovering + 4h trend bullish (pullback entry)
        elif rsi_recovering and trend_4h_bullish and trend_1d_bullish:
            desired_signal = REDUCED_SIZE
        # Fallback: Extreme oversold alone (ensures trades on all symbols)
        elif rsi_4h[i] < 20:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        # Primary: RSI overbought + aligned with 1d trend
        if rsi_overbought and trend_1d_bearish:
            desired_signal = -BASE_SIZE
        # Secondary: RSI weakening + 4h trend bearish (pullback entry)
        elif rsi_weakening and trend_4h_bearish and trend_1d_bearish:
            if desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        # Fallback: Extreme overbought alone (ensures trades on all symbols)
        elif rsi_4h[i] > 80 and desired_signal == 0:
            desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        # This prevents premature exits on minor pullbacks
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend intact and RSI not overbought
                if trend_1d_bullish and rsi_4h[i] < RSI_EXIT_LONG:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and RSI not oversold
                if trend_1d_bearish and rsi_4h[i] > RSI_EXIT_SHORT:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses
            if trend_1d_bearish and rsi_4h[i] > RSI_EXIT_LONG:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses
            if trend_1d_bullish and rsi_4h[i] < RSI_EXIT_SHORT:
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
                # Position flip
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