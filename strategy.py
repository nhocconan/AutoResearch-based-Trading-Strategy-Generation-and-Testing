#!/usr/bin/env python3
"""
Experiment #1001: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX Filter

Hypothesis: After 726 failed strategies, complex regime-switching and funding-rate
dependencies are causing 0-trade failures. KAMA (Kaufman Adaptive Moving Average)
adapts to market efficiency ratio — fast in trends, slow in chop. Combined with
ADX trend-strength filter and 1d HMA macro bias, this should generate consistent
trades across ALL symbols (BTC/ETH/SOL) without over-complication.

Why this differs from failures:
1. NO funding rate dependency (causes 0 trades when data unavailable/misaligned)
2. NO complex regime switching (CHOP index failed in exp#989-1000)
3. NO vol spike logic (too many false signals, failed exp#999)
4. SINGLE clear entry signal: KAMA crossover + ADX confirmation + HTF bias
5. KAMA adapts automatically to choppy vs trending — no manual regime detection

Key mechanics:
- KAMA(21) on 4h: adapts speed based on Efficiency Ratio (ER)
- ADX(14) > 20: confirms trend strength (avoids whipsaw entries)
- 1d HMA(21): macro bias filter (long only if price > 1d HMA, short if <)
- Entry: KAMA crosses above/below + ADX > 20 + HTF bias aligned
- Exit: KAMA reverse crossover OR 2.5x ATR trailing stoploss
- Size: 0.30 max, 0.20 reduced (discrete levels minimize fee churn)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 25-40 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_1d_hma_trend_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market Efficiency Ratio (ER).
    Fast in trends, slow in choppy markets.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(slow_period, n):
        signal = np.abs(close[i] - close[i - slow_period])
        noise = np.sum(np.abs(np.diff(close[i-slow_period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 20 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di_pct = 100 * plus_di / (atr + 1e-10)
        minus_di_pct = 100 * minus_di / (atr + 1e-10)
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di_pct[i] + minus_di_pct[i]
        if di_sum > 1e-10:
            dx[i] = 100 * np.abs(plus_di_pct[i] - minus_di_pct[i]) / di_sum
        else:
            dx[i] = 0.0
    
    # Smooth DX to get ADX
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx, plus_di_pct, minus_di_pct

def calculate_hma(series, period):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index for exit filter."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    
    # Calculate and align 1d HMA for macro bias
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
    
    # Track KAMA crossover
    prev_kama = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]):
            prev_kama = kama_4h[i] if not np.isnan(kama_4h[i]) else prev_kama
            continue
        if np.isnan(hma_1d_aligned[i]) or atr_4h[i] <= 1e-10:
            prev_kama = kama_4h[i]
            continue
        
        # === MACRO BIAS (1d HTF HMA21) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_4h[i] > 20
        trend_very_strong = adx_4h[i] > 25
        
        # === KAMA CROSSOVER DETECTION ===
        kama_cross_up = False
        kama_cross_down = False
        
        if not np.isnan(prev_kama) and not np.isnan(kama_4h[i-1]):
            # Price crosses above KAMA
            if close[i-1] < kama_4h[i-1] and close[i] > kama_4h[i]:
                kama_cross_up = True
            # Price crosses below KAMA
            if close[i-1] > kama_4h[i-1] and close[i] < kama_4h[i]:
                kama_cross_down = True
        
        # Also check KAMA slope
        kama_slope_up = False
        kama_slope_down = False
        if i > 1 and not np.isnan(kama_4h[i-1]):
            if kama_4h[i] > kama_4h[i-1] * 1.001:
                kama_slope_up = True
            if kama_4h[i] < kama_4h[i-1] * 0.999:
                kama_slope_down = True
        
        # === RSI FILTER ===
        rsi_oversold = rsi_4h[i] < 40
        rsi_overbought = rsi_4h[i] > 60
        rsi_neutral = 35 < rsi_4h[i] < 65
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: KAMA cross up + ADX strong + macro bull
        if kama_cross_up and trend_strong and macro_bull:
            desired_signal = BASE_SIZE
        # Secondary: KAMA slope up + ADX very strong + macro bull + RSI not overbought
        elif kama_slope_up and trend_very_strong and macro_bull and not rsi_overbought:
            desired_signal = REDUCED_SIZE
        # Tertiary: Price > KAMA + ADX strong + macro bull + RSI oversold (pullback entry)
        elif close[i] > kama_4h[i] and trend_strong and macro_bull and rsi_oversold:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY ===
        # Primary: KAMA cross down + ADX strong + macro bear
        if kama_cross_down and trend_strong and macro_bear:
            desired_signal = -BASE_SIZE
        # Secondary: KAMA slope down + ADX very strong + macro bear + RSI not oversold
        elif kama_slope_down and trend_very_strong and macro_bear and not rsi_oversold:
            desired_signal = -REDUCED_SIZE
        # Tertiary: Price < KAMA + ADX strong + macro bear + RSI overbought (pullback entry)
        elif close[i] < kama_4h[i] and trend_strong and macro_bear and rsi_overbought:
            desired_signal = -REDUCED_SIZE
        
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
                # Hold long if price > KAMA and ADX still strong and macro bull
                if close[i] > kama_4h[i] and adx_4h[i] > 18 and macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if price < KAMA and ADX still strong and macro bear
                if close[i] < kama_4h[i] and adx_4h[i] > 18 and macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses + price < KAMA
            if macro_bear and close[i] < kama_4h[i]:
                desired_signal = 0.0
            # Exit if RSI extremely overbought
            if rsi_4h[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses + price > KAMA
            if macro_bull and close[i] > kama_4h[i]:
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
        prev_kama = kama_4h[i]
    
    return signals