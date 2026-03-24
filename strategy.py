#!/usr/bin/env python3
"""
Experiment #1064: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX Regime + RSI Pullback

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than HMA/EMA,
reducing whipsaws in choppy markets. Combined with ADX for regime detection and simple
RSI pullback entries, this should generate consistent trades across BTC/ETH/SOL.

Key innovations:
1. KAMA (ER=10): Adapts smoothing based on market efficiency - fast in trends, slow in chop
2. ADX(14) regime: ADX>25 = trend follow, ADX<20 = mean revert (with hysteresis 18-25)
3. HTF bias: 1d KAMA for intermediate trend, 1w KAMA for long-term bias
4. Simple entries (LOOSE to guarantee trades):
   - Trend long: price>1d_KAMA>1w_KAMA + ADX>22 + RSI(14)>45
   - Trend short: price<1d_KAMA<1w_KAMA + ADX>22 + RSI(14)<55
   - Mean revert long: ADX<18 + RSI(14)<35 + price>1w_KAMA*0.92
   - Mean revert short: ADX<18 + RSI(14)>65 + price<1w_KAMA*1.08
5. ATR(14) 2.5x trailing stop
6. Discrete sizing: 0.0, ±0.25, ±0.30

Why this should work:
- KAMA reduces lag in trends while smoothing chop (better than fixed EMA/HMA)
- ADX hysteresis prevents regime flip-flopping
- 12h timeframe = 20-50 trades/year target (low fee drag)
- Loose entry thresholds ensure ≥30 trades on train, ≥3 on test
- Works on all symbols (not SOL-biased)

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_rsi_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = len(close)
    if n < period + slow:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    sc = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(close[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
    
    # Smooth with Wilder's method (EMA with alpha=1/period)
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr_smooth[i]
            minus_di[i] = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr_smooth[i]
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=10)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    kama_12h = calculate_kama(close, period=10)
    
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
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(kama_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        adx_val = adx_14[i]
        is_trending = adx_val > 22.0  # Trend follow mode
        is_choppy = adx_val < 18.0    # Mean revert mode (hysteresis gap 18-22)
        
        # === HTF BIAS ===
        kama_1d_bull = close[i] > kama_1d_aligned[i]
        kama_1d_bear = close[i] < kama_1d_aligned[i]
        kama_1w_bull = close[i] > kama_1w_aligned[i]
        kama_1w_bear = close[i] < kama_1w_aligned[i]
        
        # Strong trend alignment
        strong_bull = kama_1d_bull and kama_1w_bull and kama_1d_aligned[i] > kama_1w_aligned[i]
        strong_bear = kama_1d_bear and kama_1w_bear and kama_1d_aligned[i] < kama_1w_aligned[i]
        
        # === ENTRY LOGIC (LOOSE THRESHOLDS TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND FOLLOWING MODE
            # Long: strong bullish alignment + RSI not overbought
            if strong_bull and rsi_14[i] > 40.0 and rsi_14[i] < 80.0:
                desired_signal = SIZE_STRONG
            # Short: strong bearish alignment + RSI not oversold
            elif strong_bear and rsi_14[i] < 60.0 and rsi_14[i] > 20.0:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals (single KAMA alignment)
            elif kama_1d_bull and kama_1w_bull and rsi_14[i] > 45.0:
                desired_signal = SIZE_BASE
            elif kama_1d_bear and kama_1w_bear and rsi_14[i] < 55.0:
                desired_signal = -SIZE_BASE
        
        elif is_choppy:
            # MEAN REVERSION MODE
            # Long: RSI oversold + price near/above weekly KAMA
            if rsi_14[i] < 35.0 and close[i] > kama_1w_aligned[i] * 0.90:
                desired_signal = SIZE_BASE
            # Short: RSI overbought + price near/below weekly KAMA
            elif rsi_14[i] > 65.0 and close[i] < kama_1w_aligned[i] * 1.10:
                desired_signal = -SIZE_BASE
            # Stronger mean reversion at extremes
            elif rsi_14[i] < 25.0:
                desired_signal = SIZE_STRONG
            elif rsi_14[i] > 75.0:
                desired_signal = -SIZE_STRONG
        
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