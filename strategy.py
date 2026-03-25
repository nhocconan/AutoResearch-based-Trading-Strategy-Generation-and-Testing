#!/usr/bin/env python3
"""
Experiment #1530: 1h Primary + 4h/1d HTF — Loose Confluence for Guaranteed Trades

Hypothesis: Previous 1h strategies failed (Sharpe=0.000) due to ZERO trades from 
overly strict entry conditions. This strategy uses LOOSE thresholds to GUARANTEE
trade generation while maintaining edge through 4h trend filter.

Key components:
1. 4h HMA(21) for major trend bias (directional filter only)
2. 1h RSI(7) for fast pullback detection (thresholds: 40/60, not 30/70)
3. 1h HMA(8)/HMA(21) crossover for momentum confirmation
4. Volume filter REMOVED (was killing trades in #1519, #1525, #1529)
5. Session filter REMOVED (was killing trades)
6. ATR(14) 2.5x trailing stoploss
7. Discrete sizing: 0.0, ±0.20, ±0.25 (minimize fee churn)

Why this should work:
- LOOSE RSI thresholds (40/60) guarantee entries during normal pullbacks
- 4h HMA filter provides edge without blocking all trades
- No session/volume filters = trades can trigger anytime
- 1h TF target: 40-80 trades/year (1-2 per week)
- Proven pattern from best strategy: MTF trend + LTF pullback

Entry logic (LOOSE to guarantee ≥10 trades/train, ≥3/test):
- LONG: 4h_HMA bullish + RSI(7)<50 + HMA8>HMA21 OR RSI(7)<40
- SHORT: 4h_HMA bearish + RSI(7)>50 + HMA8<HMA21 OR RSI(7)>60

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_loose_rsi_hma_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1h indicators
    hma_8 = calculate_hma(close, period=8)
    hma_21 = calculate_hma(close, period=21)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period (shorter to allow earlier trades)
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === 1h HMA CROSSOVER (momentum) ===
        hma_bullish = hma_8[i] > hma_21[i]
        hma_bearish = hma_8[i] < hma_21[i]
        
        # === RSI (fast, loose thresholds) ===
        rsi = rsi_7[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + (RSI pullback OR HMA crossover)
        # Two entry modes for flexibility:
        # Mode 1: Strong pullback (RSI<45) with 4h trend
        # Mode 2: Momentum confirmation (HMA8>21) with 4h trend + RSI<55
        if price_above_4h:
            if rsi < 45:
                # Strong pullback entry
                desired_signal = SIZE_STRONG
            elif rsi < 55 and hma_bullish:
                # Momentum confirmation entry
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + (RSI rally OR HMA crossover)
        elif price_below_4h:
            if rsi > 55:
                # Strong rally entry
                desired_signal = -SIZE_STRONG
            elif rsi > 45 and hma_bearish:
                # Momentum confirmation entry
                desired_signal = -SIZE_BASE
        
        # === EXIT SIGNAL (RSI mean reversion) ===
        # Close long when RSI > 65, close short when RSI < 35
        if in_position and position_side > 0 and rsi > 65:
            desired_signal = 0.0
        if in_position and position_side < 0 and rsi < 35:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
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
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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