#!/usr/bin/env python3
"""
Experiment #1422: 4h Primary + 1d HTF — KAMA Adaptive Trend + ROC Momentum + RSI Pullback

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA.
During high volatility (crashes), KAMA slows down and reduces whipsaw. During trends,
KAMA speeds up and captures moves. Combined with:
1. 1d HMA(21) for major trend bias (avoid counter-trend in crashes like 2022)
2. 4h KAMA(14) vs KAMA(42) crossover for adaptive trend momentum
3. ROC(10) momentum confirmation (ROC > 0 for long, < 0 for short)
4. RSI(14) pullback entry in 40-60 zone (LOOSE to guarantee trades)
5. ATR(14) trailing stoploss at 2.5x (signal→0 when stopped)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should beat dual HMA (Sharpe=0.575):
- KAMA adapts to volatility regime (better in 2022 crash than fixed HMA)
- ROC momentum filter reduces false breakouts
- Same loose RSI entry ensures trade generation
- 4h TF = natural 25-45 trades/year (fee-efficient)

Entry logic (LOOSE to guarantee trades):
- LONG: 1d_HMA bullish + 4h_KAMA14 > 4h_KAMA42 + ROC > 0 + RSI > 40
- SHORT: 1d_HMA bearish + 4h_KAMA14 < 4h_KAMA42 + ROC < 0 + RSI < 60

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_roc_rsi_pullback_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    change = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i - period]):
            change[i] = abs(close[i] - close[i - period])
        
        vol_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0 and not np.isnan(close[j]) and not np.isnan(close[j - 1]):
                vol_sum += abs(close[j] - close[j - 1])
        volatility[i] = vol_sum
    
    er = np.zeros(n)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = sc ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(close[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0 and not np.isnan(close[i]) and not np.isnan(close[i - period]):
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_kama(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    kama_14 = calculate_kama(close, period=14)
    kama_42 = calculate_kama(close, period=42)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    roc_10 = calculate_roc(close, period=10)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_14[i]) or np.isnan(kama_42[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(roc_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d KAMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 4h KAMA CROSSOVER (adaptive trend momentum) ===
        kama_bullish = kama_14[i] > kama_42[i]
        kama_bearish = kama_14[i] < kama_42[i]
        
        # === ROC MOMENTUM (confirm direction) ===
        roc_positive = roc_10[i] > 0
        roc_negative = roc_10[i] < 0
        
        # === RSI PULLBACK (LOOSE entry - guarantee trades) ===
        rsi = rsi_14[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1d bullish + 4h KAMA bullish + ROC positive + RSI > 40
        if price_above_1d and kama_bullish and roc_positive and rsi > 40:
            # Strong if RSI also < 70 (not overbought)
            if rsi < 70:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        # SHORT: 1d bearish + 4h KAMA bearish + ROC negative + RSI < 60
        elif price_below_1d and kama_bearish and roc_negative and rsi < 60:
            # Strong if RSI also > 30 (not oversold)
            if rsi > 30:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
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