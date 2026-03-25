#!/usr/bin/env python3
"""
Experiment #1342: 4h Primary + 1d/1w HTF — HMA Trend + RSI Pullback + Weekly Regime

Hypothesis: 4h timeframe with multi-HTF filtering will achieve better risk-adjusted returns
than 6h strategies. Key innovations vs failed experiments:

1. 1w HMA(21) for MAJOR regime bias (only trade with weekly trend direction)
2. 1d HMA(21) for intermediate trend confirmation (aligns with weekly)
3. 4h HMA(16/48) crossover for entry timing (faster than 21)
4. RSI(7) pullback entries (not extremes - enter at 45-55 zone during trend)
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry conditions to guarantee 20-50 trades/year on 4h

Why this should work (learning from failures):
- 4h = natural 20-50 trades/year (fee-friendly, proven in research)
- Triple HTF filter (1w+1d+4h) = strong directional bias without over-filtering
- RSI pullback (not extreme) = catches trend continuations, not reversals
- HMA crossover (16/48) = faster signals than single HMA, less lag
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn
- Stoploss via signal=0 = proper risk management

Entry logic (LOOSE to guarantee trades):
- LONG: 1w_HMA bullish + 1d_HMA rising + 4h_HMA16>48 + RSI(7) 40-60
- SHORT: 1w_HMA bearish + 1d_HMA falling + 4h_HMA16<48 + RSI(7) 40-60

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_weekly_regime_1d1w_v1"
timeframe = "4h"
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

def calculate_rsi(close, period=7):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    
    # 4h HMA crossover (fast/slow)
    hma_4h_fast = calculate_hma(close, period=16)
    hma_4h_slow = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_fast[i]) or np.isnan(hma_4h_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === WEEKLY REGIME BIAS (1w HMA) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND CONFIRMATION (1d HMA slope) ===
        hma_1d_slope = 0.0
        if i >= 3 and not np.isnan(hma_1d_aligned[i-3]):
            hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-3]
        
        daily_bullish = hma_1d_slope > 0
        daily_bearish = hma_1d_slope < 0
        
        # === 4h HMA CROSSOVER ===
        hma_cross_bullish = hma_4h_fast[i] > hma_4h_slow[i]
        hma_cross_bearish = hma_4h_fast[i] < hma_4h_slow[i]
        
        # === RSI PULLBACK ZONE (not extremes) ===
        rsi = rsi_7[i]
        rsi_neutral = 40.0 <= rsi <= 60.0
        rsi_bullish_pullback = 40.0 <= rsi <= 55.0
        rsi_bearish_pullback = 45.0 <= rsi <= 60.0
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Weekly bullish + Daily rising + 4h HMA cross + RSI pullback
        if price_above_1w and daily_bullish and hma_cross_bullish:
            if rsi_bullish_pullback:
                if rsi <= 50.0:
                    desired_signal = SIZE_STRONG  # Deeper pullback
                else:
                    desired_signal = SIZE_BASE  # Shallow pullback
        
        # SHORT: Weekly bearish + Daily falling + 4h HMA cross + RSI pullback
        elif price_below_1w and daily_bearish and hma_cross_bearish:
            if rsi_bearish_pullback:
                if rsi >= 50.0:
                    desired_signal = -SIZE_STRONG  # Higher pullback
                else:
                    desired_signal = -SIZE_BASE  # Lower pullback
        
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