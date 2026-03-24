#!/usr/bin/env python3
"""
Experiment #789: 15m Primary + 1h/1d HTF — Fast Mean Reversion with HTF Bias

Hypothesis: 15m timeframe with 1h trend bias can capture intraday mean reversion
while respecting higher-timeframe direction. Previous 15m experiments failed with
Sharpe=0.000 (ZERO trades) due to overly strict session filters and confluence
requirements. This version uses LOOSE thresholds to guarantee trade generation.

Key innovations:
1. 1h HMA(21) for HTF trend bias — direction filter only
2. 15m RSI(7) with loose thresholds (40/60 not 30/70) for frequent entries
3. 15m HMA(8/21) crossover for momentum confirmation (alternative entry)
4. NO session filter — was blocking 80%+ of potential trades
5. ATR(14) 2.0x trailing stop for tight risk management
6. Discrete sizing: 0.0, ±0.15, ±0.20 (smaller for 15m frequency)
7. 1d HMA as meta-filter for extreme regimes (optional override)

Entry conditions (LOOSE to ensure ≥40 trades/year):
- LONG: 1h HMA bull + (RSI<45 OR 15m HMA bull crossover)
- SHORT: 1h HMA bear + (RSI>55 OR 15m HMA bear crossover)
- 1d HMA extreme: override if price >2% from 1d HMA (mean reversion)

Target: Sharpe>0.40, trades>=40/year, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_hma_fast_1h1d_loose_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
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
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_8 = calculate_hma(close, period=8)
    hma_21 = calculate_hma(close, period=21)
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]) or np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1h HMA) ===
        htf_1h_bull = close[i] > hma_1h_aligned[i]
        htf_1h_bear = close[i] < hma_1h_aligned[i]
        
        # === 1d EXTREME REGIME (optional mean reversion override) ===
        hma_1d_val = hma_1d_aligned[i]
        extreme_long = False
        extreme_short = False
        if not np.isnan(hma_1d_val) and hma_1d_val > 1e-10:
            pct_from_1d = (close[i] - hma_1d_val) / hma_1d_val
            extreme_long = pct_from_1d < -0.02  # Price 2% below 1d HMA
            extreme_short = pct_from_1d > 0.02  # Price 2% above 1d HMA
        
        # === 15m HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_8[i-1]) and not np.isnan(hma_21[i-1]):
            hma_crossover_long = (hma_8[i-1] <= hma_21[i-1]) and (hma_8[i] > hma_21[i])
            hma_crossover_short = (hma_8[i-1] >= hma_21[i-1]) and (hma_8[i] < hma_21[i])
        
        # === 15m HMA TREND ===
        hma_15m_bull = hma_8[i] > hma_21[i]
        hma_15m_bear = hma_8[i] < hma_21[i]
        
        # === RSI CONDITIONS (LOOSE for more trades) ===
        rsi_oversold = rsi_7[i] < 45.0
        rsi_overbought = rsi_7[i] > 55.0
        rsi_extreme_oversold = rsi_7[i] < 30.0
        rsi_extreme_overbought = rsi_7[i] > 70.0
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + (RSI oversold OR HMA crossover OR HMA bull)
        if htf_1h_bull:
            if rsi_oversold or hma_crossover_long or hma_15m_bull:
                if rsi_extreme_oversold or hma_crossover_long or extreme_long:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + (RSI overbought OR HMA crossover OR HMA bear)
        elif htf_1h_bear:
            if rsi_overbought or hma_crossover_short or hma_15m_bear:
                if rsi_extreme_overbought or hma_crossover_short or extreme_short:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === 1d EXTREME OVERRIDE (mean reversion signal) ===
        # If price is extremely far from 1d HMA, fade regardless of 1h bias
        if extreme_long and desired_signal <= 0:
            desired_signal = SIZE_BASE
        if extreme_short and desired_signal >= 0:
            desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals