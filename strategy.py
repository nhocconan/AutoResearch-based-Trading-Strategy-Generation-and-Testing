#!/usr/bin/env python3
"""
Experiment #156: 12h Primary + 1d HTF — HMA Crossover + RSI Pullback + Volume Filter

Hypothesis: After analyzing 155 failed experiments, the pattern is clear:
- Complex regime switching (Choppiness + CRSI + Fisher) often produces 0 trades
- Simpler HMA crossover + RSI pullback has historically worked (SOL +0.879)
- 12h timeframe needs LOOSE filters to ensure >=30 trades on train, >=3 on test
- 1d HMA(50) provides major trend bias without being too restrictive
- Volume filter ensures we only trade on meaningful moves (avoid chop)
- This is SIMPLER than #142 but should generate MORE reliable trades

Key design choices:
- Timeframe: 12h (20-50 trades/year target)
- HTF: 1d HMA(50) for major trend bias (call ONCE before loop)
- Entry: HMA(21/63) crossover + RSI(14) pullback + Volume > SMA(20)
- Position size: 0.30 (30% of capital, conservative)
- Stoploss: 2.5x ATR(14) trailing
- LOOSE RSI filter (25-75) to ensure trades generate on all symbols

Target: Sharpe>0.375 (beat #152), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crossover_rsi_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_fast = calculate_hma(close, period=21)
    hma_slow = calculate_hma(close, period=63)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_sma[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === HMA CROSSOVER SIGNAL ===
        hma_crossover_bull = hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]
        hma_crossover_bear = hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]
        
        # === HMA TREND STATE (not just crossover) ===
        hma_trend_bull = hma_fast[i] > hma_slow[i]
        hma_trend_bear = hma_fast[i] < hma_slow[i]
        
        # === VOLUME FILTER (above average) ===
        vol_ok = volume[i] > vol_sma[i] * 0.8  # loose filter
        
        # === RSI PULLBACK FILTER (LOOSE to ensure trades) ===
        # Long: RSI pulled back but still bullish (30-60)
        rsi_pullback_long = 30.0 < rsi[i] < 65.0
        # Short: RSI bounced but still bearish (40-70)
        rsi_pullback_short = 35.0 < rsi[i] < 70.0
        
        # === RSI EXTREMES (mean reversion entries) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG entries (multiple conditions to ensure trades generate)
        if hma_trend_bull and htf_bull and vol_ok:
            # Primary: HMA crossover + RSI pullback
            if hma_crossover_bull and rsi_pullback_long:
                desired_signal = SIZE
            # Secondary: RSI oversold in uptrend
            elif rsi_oversold and rsi[i] > 25.0:
                desired_signal = SIZE * 0.7
            # Tertiary: Strong trend continuation
            elif hma_trend_bull and rsi[i] > 45.0 and rsi[i] < 60.0:
                desired_signal = SIZE * 0.5
        
        # SHORT entries
        elif hma_trend_bear and htf_bear and vol_ok:
            # Primary: HMA crossover + RSI pullback
            if hma_crossover_bear and rsi_pullback_short:
                desired_signal = -SIZE
            # Secondary: RSI overbought in downtrend
            elif rsi_overbought and rsi[i] < 75.0:
                desired_signal = -SIZE * 0.7
            # Tertiary: Strong trend continuation
            elif hma_trend_bear and rsi[i] > 40.0 and rsi[i] < 55.0:
                desired_signal = -SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
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