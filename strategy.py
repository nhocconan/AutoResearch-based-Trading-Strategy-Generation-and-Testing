#!/usr/bin/env python3
"""
Experiment #089: 4h Primary + 1d HTF — KAMA Adaptive Trend + RSI Filter

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better
than HMA, reducing whipsaws in ranging markets while capturing trends efficiently.
Combined with loose RSI filter and 1d trend bias, this should generate 20-50 trades/year
on 4h timeframe with better risk-adjusted returns than simple HMA crossover.

Key design choices:
- Timeframe: 4h (proven to work well, 20-50 trades/year target)
- HTF: 1d HMA(50) for major trend bias only (simpler, more reliable)
- Primary: KAMA(10,2,30) crossover with signal line for adaptive entries
- RSI thresholds: 40/60 (loose enough to ensure trade generation)
- Position size: 0.25 (25% of capital, conservative for 4h)
- Stoploss: 2.5x ATR trailing (tighter than 3x for better risk control)

Why KAMA over HMA:
- KAMA adapts smoothing based on market efficiency ratio
- Less whipsaw in choppy conditions (common in 2022, 2025 bear markets)
- Faster response in trending conditions
- Proven in quantitative literature for crypto volatility regimes

Target: Beat Sharpe=0.351 (current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_adaptive_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market volatility/efficiency
    period: lookback for efficiency ratio
    fast_period: fastest smoothing constant (default 2)
    slow_period: slowest smoothing constant (default 30)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    # ER = |close - close[period]| / sum(|close[i] - close[i-1]|)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(i - period + 1, i + 1):
            volatility += abs(close[j] - close[j - 1])
        
        if volatility < 1e-10:
            er = 0.0
        else:
            er = price_change / volatility
        
        # Calculate Smoothing Constant (SC)
        # SC = [ER * (fast - slow) + slow]^2
        fast_sc = 2.0 / (fast_period + 1.0)
        slow_sc = 2.0 / (slow_period + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_kama_signal(kama, period=10, fast_period=2, slow_period=30):
    """KAMA signal line (KAMA of KAMA) for crossover detection"""
    signal = calculate_kama(kama, period, fast_period, slow_period)
    return signal

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_signal = calculate_kama_signal(kama, period=10, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 4h)
    
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
        if np.isnan(kama[i]) or np.isnan(kama_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        # Simple: is price above or below daily HMA?
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h ADAPTIVE TREND (KAMA crossover) ===
        kama_cross_bull = kama[i] > kama_signal[i]
        kama_cross_bear = kama[i] < kama_signal[i]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        # For longs: RSI > 40 (not oversold)
        # For shorts: RSI < 60 (not overbought)
        rsi_ok_long = rsi[i] > 40.0
        rsi_ok_short = rsi[i] < 60.0
        
        # === DESIRED SIGNAL ===
        # LONG: 1d bull + 4h KAMA cross bull + RSI > 40
        # SHORT: 1d bear + 4h KAMA cross bear + RSI < 60
        desired_signal = 0.0
        
        if htf_bull and kama_cross_bull and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and kama_cross_bear and rsi_ok_short:
            desired_signal = -SIZE
        
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