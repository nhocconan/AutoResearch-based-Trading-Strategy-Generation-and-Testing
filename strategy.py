#!/usr/bin/env python3
"""
Experiment #1232: 12h Primary + 1d HTF — KAMA Adaptive Trend + ADX Regime + RSI Entry

Hypothesis: After 1000+ failed experiments, the key insight is that ADAPTIVE indicators
outperform static ones in crypto's varying volatility regimes. KAMA (Kaufman Adaptive MA)
adjusts speed based on market efficiency ratio - fast in trends, slow in chop.

Combined with ADX hysteresis (enter 25, exit 18) to avoid whipsaw, and fast RSI(7) for
12h entry timing. This should generate 25-45 trades/year on 12h timeframe.

Key differences from failed attempts:
- KAMA instead of HMA/EMA (adapts to volatility automatically)
- ADX hysteresis band (20/25) prevents rapid flip-flopping
- RSI(7) instead of RSI(14) for faster 12h signals
- LOOSE RSI range (35-65) to guarantee trades
- Discrete sizing (0.0, ±0.25, ±0.30) to minimize fee churn

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_regime_rsi7_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    kama = np.full(n, np.nan, dtype=np.float64)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio (ER)
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        if price_change > 0:
            volatility = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
            if volatility > 0:
                er[i] = price_change / volatility
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    adx = np.full(n, np.nan, dtype=np.float64)
    
    if n < period * 3:
        return adx
    
    # True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed averages
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    plus_di = np.divide(plus_di, atr, out=np.zeros_like(plus_di), where=atr > 0) * 100
    minus_di = np.divide(minus_di, atr, out=np.zeros_like(minus_di), where=atr > 0) * 100
    
    # DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    di_sum = plus_di + minus_di
    for i in range(period * 2, n):
        if di_sum[i] > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum[i]
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2:] = adx_raw[period * 2:]
    
    return adx

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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)  # Faster RSI for 12h entries
    
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
    
    # ADX hysteresis state
    adx_above_entry = False  # Track if ADX crossed above 25
    
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
        
        if np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily KAMA) ===
        price_above_1d = close[i] > kama_1d_aligned[i]
        price_below_1d = close[i] < kama_1d_aligned[i]
        
        # === ADX REGIME WITH HYSTERESIS ===
        adx = adx_14[i]
        
        # ADX hysteresis: enter when >25, stay in trend until <18
        if adx > 25.0:
            adx_above_entry = True
        elif adx < 18.0:
            adx_above_entry = False
        
        trend_regime = adx_above_entry  # True = trend mode, False = chop mode
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        rsi = rsi_7[i]
        
        # In trend regime (ADX hysteresis active)
        if trend_regime:
            # LONG: Price above 1d KAMA + RSI pullback (35-65 range)
            if price_above_1d and 35.0 <= rsi <= 65.0:
                # Strong if RSI rising
                if i > 0 and rsi_7[i] > rsi_7[i-1]:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
            
            # SHORT: Price below 1d KAMA + RSI pullback (35-65 range)
            elif price_below_1d and 35.0 <= rsi <= 65.0:
                # Strong if RSI falling
                if i > 0 and rsi_7[i] < rsi_7[i-1]:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        else:
            # In chop regime - smaller positions, tighter RSI range
            if price_above_1d and 40.0 <= rsi <= 60.0:
                desired_signal = SIZE_BASE * 0.6  # Reduced size in chop
            elif price_below_1d and 40.0 <= rsi <= 60.0:
                desired_signal = -SIZE_BASE * 0.6  # Reduced size in chop
        
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
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            # Small position in chop
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.6
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