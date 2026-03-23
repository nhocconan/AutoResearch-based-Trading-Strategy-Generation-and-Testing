#!/usr/bin/env python3
"""
Experiment #1079: 4h Primary + 1d HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After analyzing 781+ failed experiments, the winning pattern for 4h combines:
1. KAMA (Kaufman Adaptive Moving Average) — adapts speed to market efficiency
   Fast in trends, slow in ranges. Better than EMA/HMA for crypto whipsaws.
   Long: KAMA(10) crosses above KAMA(40) | Short: KAMA(10) crosses below KAMA(40)
2. ADX(14) — trend strength filter (not direction)
   ADX > 20 = trend valid | ADX < 20 = range (reduce position or skip)
3. Choppiness Index — regime confirmation
   CHOP > 61.8 = reduce size by 50% (range market)
   CHOP < 38.2 = full size (trending market)
4. 1d KAMA21 macro bias — only trade in direction of daily adaptive trend
5. ATR(14) trailing stop — 2.5x ATR from entry/extreme

Why this should beat Sharpe=0.612:
- KAMA is PROVEN adaptive indicator (different from all failed RSI/CRSI/Fisher/STC strategies)
- ADX filter prevents entries in weak trends (major failure mode)
- Simpler logic = more consistent signals across BTC/ETH/SOL
- 4h timeframe = 30-50 trades/year target (optimal fee/trade balance)
- Less complex than #1071 STC version = fewer bugs, more trades

Timeframe: 4h (primary)
HTF: 1d (daily) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels (reduced to 0.15 in choppy markets)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_chop_1d_kama_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    
    Formula:
    1. Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    2. Smoothing Constant (SC) = [ER * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)]^2
    3. KAMA[i] = KAMA[i-1] + SC * (Close[i] - KAMA[i-1])
    
    Adapts to market noise: fast in trends, slow in ranges.
    Proven in crypto for reducing whipsaw vs EMA/SMA.
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period - 1] = close[period - 1]
    for i in range(period, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) — measures trend strength (not direction).
    
    Formula:
    1. Calculate +DM, -DM, TR
    2. Smooth +DM, -DM, TR over period (Wilder's smoothing)
    3. +DI = 100 * +DM / TR | -DI = 100 * -DM / TR
    4. DX = 100 * |+DI - -DI| / (+DI + -DI)
    5. ADX = SMA(DX, period)
    
    ADX > 25 = strong trend | ADX < 20 = range/chop
    """
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx, plus_di, minus_di
    
    # Calculate True Range, +DM, -DM
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i] and high[i] - high[i-1] > 0:
            plus_dm[i] = high[i] - high[i-1]
        if low[i-1] - low[i] > high[i] - high[i-1] and low[i-1] - low[i] > 0:
            minus_dm[i] = low[i-1] - low[i]
    
    # Wilder's smoothing (RMA)
    def rma(series, period):
        result = np.full(len(series), np.nan)
        result[period - 1] = np.nansum(series[:period]) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period - 1) + series[i]) / period
        return result
    
    tr_smooth = rma(tr, period)
    plus_dm_smooth = rma(plus_dm, period)
    minus_dm_smooth = rma(minus_dm, period)
    
    # Calculate DI
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0.0
    
    adx = rma(dx, period)
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) — measures market choppiness vs trending.
    
    Formula:
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8 = choppy/range market
    - CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    log_period = np.log10(period)
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / log_period
        else:
            chop[i] = 50.0
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
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
    
    # Calculate and align 1d KAMA21 for macro trend filter
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=21, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (4h) indicators
    kama_fast = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, period=40, fast_period=2, slow_period=30)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15  # Half size in choppy markets
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_fast[i]) or np.isnan(kama_slow[i]):
            continue
        if np.isnan(adx[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === VOLATILITY/REGIME (Position Sizing) ===
        is_choppy = chop[i] > 61.8
        current_size = REDUCED_SIZE if is_choppy else BASE_SIZE
        
        # === MACRO TREND (1d KAMA21) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === TREND STRENGTH (ADX) ===
        trend_valid = adx[i] > 20.0
        
        # === KAMA CROSSOVER SIGNALS ===
        kama_bull_cross = kama_fast[i-1] <= kama_slow[i-1] and kama_fast[i] > kama_slow[i]
        kama_bear_cross = kama_fast[i-1] >= kama_slow[i-1] and kama_fast[i] < kama_slow[i]
        
        # Current KAMA alignment
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # === DI CONFIRMATION ===
        di_bull = plus_di[i] > minus_di[i]
        di_bear = minus_di[i] > plus_di[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: KAMA bull cross + ADX valid + macro bull + DI bull
        if kama_bull_cross and trend_valid and macro_bull and di_bull:
            desired_signal = current_size
        # Secondary: KAMA bullish + ADX valid + macro bull (no cross, holding)
        elif kama_bullish and trend_valid and macro_bull and di_bull and not in_position:
            desired_signal = current_size * 0.5
        
        # === SHORT ENTRY ===
        # Primary: KAMA bear cross + ADX valid + macro bear + DI bear
        elif kama_bear_cross and trend_valid and macro_bear and di_bear:
            desired_signal = -current_size
        # Secondary: KAMA bearish + ADX valid + macro bear (no cross, holding)
        elif kama_bearish and trend_valid and macro_bear and di_bear and not in_position:
            desired_signal = -current_size * 0.5
        
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
                # Hold long if KAMA still bullish and ADX valid
                if kama_bullish and (adx[i] > 15.0 or macro_bull):
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if KAMA still bearish and ADX valid
                if kama_bearish and (adx[i] > 15.0 or macro_bear):
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA crosses bearish OR macro reverses
            if kama_bear_cross:
                desired_signal = 0.0
            elif macro_bear and adx[i] > 25.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA crosses bullish OR macro reverses
            if kama_bull_cross:
                desired_signal = 0.0
            elif macro_bull and adx[i] > 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals