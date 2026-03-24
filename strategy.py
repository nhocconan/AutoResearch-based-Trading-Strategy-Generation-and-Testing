#!/usr/bin/env python3
"""
Experiment #004: 12h Primary + 1d HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After analyzing failed experiments, the pattern for 12h timeframe shows:
- Pure trend following (EMA/HMA crossover) fails on BTC/ETH in bear markets (2022, 2025)
- Pure mean reversion fails on SOL during strong trends
- KAMA (Kaufman Adaptive Moving Average) adapts to volatility - faster in trends, slower in chop
- ADX(14) > 20 confirms trend strength, ADX < 20 = range
- Choppiness Index confirms regime: CHOP > 55 = range, CHOP < 45 = trend
- 1d HMA(50) provides major trend bias without being too restrictive
- Dual regime: trend-follow when ADX>20+CHOP<45, mean-revert when ADX<20+CHOP>55
- LOOSE RSI filters (25-75) ensure sufficient trades on all symbols

Key design choices:
- Timeframe: 12h (target 20-50 trades/year, lower fee drag than lower TF)
- HTF: 1d HMA(50) for major trend bias (call ONCE before loop)
- Primary: KAMA(21) adaptive trend + ADX(14) strength + CHOP(14) regime
- Entry: KAMA crossover + ADX confirmation + HTF bias + RSI filter
- Regime switch: trend-follow vs mean-revert based on ADX+CHOP
- Position size: 0.28 (28% of capital, conservative for 12h swings)
- Stoploss: 2.5x ATR(14) trailing stop
- LOOSE filters to ensure >=30 trades train, >=3 test on ALL symbols

Target: Sharpe>0.4, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_chop_regime_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=21, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market volatility - moves fast in trends, slow in chop
    ER (Efficiency Ratio) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        
        if noise > 1e-10:
            er = signal / noise
        else:
            er = 0.0
        
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        if i == period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend, ADX < 20 = weak/range
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0.0)
        else:
            plus_dm[i] = 0.0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0.0)
        else:
            minus_dm[i] = 0.0
    
    # Smooth DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if tr_s[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_s[i] / tr_s[i]
            minus_di[i] = 100.0 * minus_dm_s[i] / tr_s[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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
    rsi[:] = np.nan
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
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
    
    # Calculate primary (12h) indicators
    kama = calculate_kama(close, period=21)
    adx = calculate_adx(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # KAMA fast line for crossover
    kama_fast = calculate_kama(close, period=10)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 12h)
    
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
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(kama_fast[i]):
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
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (ADX + Choppiness) ===
        # ADX > 20 + CHOP < 45 = trending regime
        # ADX < 20 OR CHOP > 55 = range/choppy regime
        is_trending = adx[i] > 20.0 and chop[i] < 45.0
        is_choppy = adx[i] < 20.0 or chop[i] > 55.0
        
        # === KAMA TREND SIGNALS ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # KAMA crossover signals
        kama_cross_bull = kama_fast[i] > kama[i] and kama_fast[i-1] <= kama[i-1]
        kama_cross_bear = kama_fast[i] < kama[i] and kama_fast[i-1] >= kama[i-1]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        rsi_ok_long = rsi[i] > 25.0
        rsi_ok_short = rsi[i] < 75.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow KAMA direction with HTF bias
            # LONG: KAMA bull + HTF bull + RSI ok + ADX confirming
            if kama_bull and htf_bull and rsi_ok_long and adx[i] > 18.0:
                desired_signal = SIZE
            # SHORT: KAMA bear + HTF bear + RSI ok + ADX confirming
            elif kama_bear and htf_bear and rsi_ok_short and adx[i] > 18.0:
                desired_signal = -SIZE
            # KAMA crossover entry (stronger signal)
            elif kama_cross_bull and htf_bull and rsi[i] > 30.0:
                desired_signal = SIZE
            elif kama_cross_bear and htf_bear and rsi[i] < 70.0:
                desired_signal = -SIZE
            # Fallback: strong KAMA trend (ignore HTF if very strong)
            elif kama_bull and rsi[i] > 35.0 and adx[i] > 25.0:
                desired_signal = SIZE * 0.7
            elif kama_bear and rsi[i] < 65.0 and adx[i] > 25.0:
                desired_signal = -SIZE * 0.7
        else:
            # CHOPPY REGIME: Mean revert at KAMA extremes
            # LONG: price far below KAMA + RSI oversold + HTF not strongly bear
            dist_below_kama = (kama[i] - close[i]) / (kama[i] + 1e-10)
            dist_above_kama = (close[i] - kama[i]) / (kama[i] + 1e-10)
            
            if dist_below_kama > 0.02 and rsi_oversold and not htf_bear:
                desired_signal = SIZE
            elif dist_above_kama > 0.02 and rsi_overbought and not htf_bull:
                desired_signal = -SIZE
            # Fallback: extreme RSI mean reversion
            elif rsi[i] < 28.0 and kama_bull:
                desired_signal = SIZE * 0.7
            elif rsi[i] > 72.0 and kama_bear:
                desired_signal = -SIZE * 0.7
        
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
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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