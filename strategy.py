#!/usr/bin/env python3
"""
Experiment #706: 1d Primary + 1w HTF — Dual Regime (Trend + Mean Reversion)

Hypothesis: Daily timeframe with weekly bias provides optimal signal quality for crypto.
Using regime detection (Choppiness Index) to switch between trend-following and mean-reversion.
1w HMA provides directional bias, daily indicators provide entry timing.

Key innovations:
1. CHOP(14) regime switch: >50 = range (mean revert), <50 = trend (trend follow)
2. RSI(7) for mean reversion entries (faster than RSI14 for daily)
3. HMA(21/63) for trend detection (63 = ~2 months on daily)
4. 1w HMA(21) for directional bias filter
5. ATR(14) 3x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.15, ±0.25, ±0.30 to minimize fee churn

Entry conditions (relaxed for trade generation):
- LONG trend: 1w HMA bull + HMA21>63 (any CHOP)
- LONG mean-revert: CHOP>50 + RSI<35 + 1w HMA bull
- SHORT trend: 1w HMA bear + HMA21<63 (any CHOP)
- SHORT mean-revert: CHOP>50 + RSI>65 + 1w HMA bear

Target: Sharpe>0.40, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 1d
Size: 0.15-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_hma_rsi_chop_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market consolidation vs trending
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    High CHOP (>61.8) = choppy/ranging, Low CHOP (<38.2) = trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            tr_sum += tr
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and tr_sum > 1e-10:
            choppiness[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50.0
    
    return choppiness

def calculate_rsi(close, period=7):
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate daily indicators
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=7)
    hma_21 = calculate_hma(close, period=21)
    hma_63 = calculate_hma(close, period=63)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (discrete levels per Rule 4)
    SIZE_STRONG = 0.30
    SIZE_BASE = 0.25
    SIZE_WEAK = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_63[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        is_choppy = chop[i] > 50.0  # Range/mean-reversion regime
        is_trending = chop[i] < 50.0  # Trend-following regime
        
        # === WEEKLY BIAS ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND ===
        hma_bull = hma_21[i] > hma_63[i]
        hma_bear = hma_21[i] < hma_63[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        signal_strength = 0  # 0=none, 1=weak, 2=base, 3=strong
        
        # LONG ENTRIES
        if weekly_bull:
            # Trend-following: HMA crossover (works in any regime)
            if hma_bull:
                signal_strength = max(signal_strength, 2)  # base size
            # Mean-reversion: RSI oversold in choppy market
            if is_choppy and rsi[i] < 35.0:
                signal_strength = max(signal_strength, 3)  # strong size
        
        # SHORT ENTRIES
        if weekly_bear:
            # Trend-following: HMA crossover (works in any regime)
            if hma_bear:
                signal_strength = max(signal_strength, 2)  # base size
            # Mean-reversion: RSI overbought in choppy market
            if is_choppy and rsi[i] > 65.0:
                signal_strength = max(signal_strength, 3)  # strong size
        
        # Weak signals: HMA alone without weekly confirmation (generates more trades)
        if hma_bull and signal_strength == 0:
            signal_strength = 1
        elif hma_bear and signal_strength == 0:
            signal_strength = -1
        
        # Convert strength to signal
        if signal_strength >= 3:
            desired_signal = SIZE_STRONG if hma_bull else -SIZE_STRONG
        elif signal_strength >= 2:
            desired_signal = SIZE_BASE if hma_bull else -SIZE_BASE
        elif signal_strength >= 1:
            desired_signal = SIZE_WEAK
        elif signal_strength <= -1:
            desired_signal = -SIZE_WEAK
        
        # === STOPLOSS CHECK (3x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
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
        elif desired_signal >= SIZE_WEAK * 0.9:
            final_signal = SIZE_WEAK
        elif desired_signal <= -SIZE_WEAK * 0.9:
            final_signal = -SIZE_WEAK
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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