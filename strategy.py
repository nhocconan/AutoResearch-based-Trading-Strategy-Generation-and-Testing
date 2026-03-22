#!/usr/bin/env python3
"""
Experiment #100: 4h KAMA Adaptive Trend + 1d HMA Bias + ATR Trailing Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility -
faster during trends, slower during chop. This should reduce whipsaws vs static EMA.
Combined with 1d HMA for higher-timeframe trend bias, we capture major moves while
avoiding counter-trend trades. ATR trailing stop protects capital during reversals.

Why this might work on 4h (learning from #088 Sharpe=0.223 and #094 Sharpe=0.000):
- #088 used Supertrend+ADX (worked but can be slow to reverse)
- #094 had too many filters = 0 trades (critical failure!)
- KAMA adapts speed based on volatility ratio (ER = Efficiency Ratio)
- Simpler entry: KAMA crossover + HTF bias only (no ADX/RSI clutter)
- ATR trailing stop ensures we exit when trend breaks
- Discrete position sizing minimizes fee churn

Timeframe: 4h (REQUIRED), HTF: 1d via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.5*ATR trailing.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_1d_hma_adaptive_trend_v1"
timeframe = "4h"
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

def calculate_kama(close, fast_period=2, slow_period=30, period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - faster in trends, slower in chop.
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    signal = np.abs(close - np.roll(close, period))
    signal[:period] = np.nan
    
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = noise[i-1] + np.abs(close[i] - close[i-1])
        if i >= period:
            noise[i] -= np.abs(close[i-period] - close[i-period-1])
    noise[:period] = np.nan
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = noise > 0
    er[mask] = signal[mask] / noise[mask]
    er[~mask] = 0.0
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    
    # KAMA fast and slow for crossover
    kama_fast = calculate_kama(close, fast_period=2, slow_period=10, period=10)
    kama_slow = calculate_kama(close, fast_period=2, slow_period=40, period=10)
    
    # Additional trend confirmation
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Track KAMA crossover state
    prev_kama_diff = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === KAMA CROSSOVER SIGNAL ===
        kama_diff = kama_fast[i] - kama_slow[i]
        
        # Detect crossover
        kama_cross_bull = prev_kama_diff <= 0 and kama_diff > 0
        kama_cross_bear = prev_kama_diff >= 0 and kama_diff < 0
        
        # Store for next iteration
        prev_kama_diff = kama_diff
        
        # === EMA ALIGNMENT (additional confirmation) ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === PRICE VS KAMA ===
        price_above_kama = close[i] > kama_slow[i]
        price_below_kama = close[i] < kama_slow[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (simplified to ensure trades) ===
        # Path 1: KAMA bull crossover + 1d bullish trend (strong signal)
        if kama_cross_bull and bull_trend_1d:
            new_signal = SIZE_STRONG
        
        # Path 2: KAMA bullish + 1d bullish + EMA bullish (trend confirmation)
        if new_signal == 0.0 and kama_bullish and bull_trend_1d and ema_bullish:
            new_signal = SIZE_BASE
        
        # Path 3: KAMA bullish + 1d bullish + price above KAMA (fallback)
        if new_signal == 0.0 and kama_bullish and bull_trend_1d and price_above_kama:
            new_signal = SIZE_BASE
        
        # Path 4: KAMA bull crossover alone (ensures we get trades)
        if new_signal == 0.0 and kama_cross_bull:
            if bull_trend_1d or ema_bullish:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (simplified to ensure trades) ===
        # Path 1: KAMA bear crossover + 1d bearish trend (strong signal)
        if kama_cross_bear and bear_trend_1d:
            new_signal = -SIZE_STRONG
        
        # Path 2: KAMA bearish + 1d bearish + EMA bearish (trend confirmation)
        if new_signal == 0.0 and kama_bearish and bear_trend_1d and ema_bearish:
            new_signal = -SIZE_BASE
        
        # Path 3: KAMA bearish + 1d bearish + price below KAMA (fallback)
        if new_signal == 0.0 and kama_bearish and bear_trend_1d and price_below_kama:
            new_signal = -SIZE_BASE
        
        # Path 4: KAMA bear crossover alone (ensures we get trades)
        if new_signal == 0.0 and kama_cross_bear:
            if bear_trend_1d or ema_bearish:
                new_signal = -SIZE_BASE
        
        # === ATR TRAILING STOPLOSS LOGIC (Rule 6) ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # === KAMA CROSSBACK EXIT ===
        # Exit long if KAMA crosses bearish
        if in_position and position_side > 0 and kama_cross_bear:
            new_signal = 0.0
        
        # Exit short if KAMA crosses bullish
        if in_position and position_side < 0 and kama_cross_bull:
            new_signal = 0.0
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals