#!/usr/bin/env python3
"""
Experiment #022: KAMA + ADX + Volume + 1d HTF Trend (4h)

HYPOTHESIS: KAMA (adaptive moving average) adapts to volatility regimes:
- Bull: KAMA rising + ADX > 20 + volume spike + 1d HTF bull = long
- Bear: KAMA falling + ADX > 20 + volume spike + 1d HTF bear = short
- Range: KAMA flat + ADX < 18 = no trade (avoid whipsaws)

WHY IT SHOULD WORK:
1. KAMA adjusts speed based on price efficiency — fast in trends, slow in ranges
2. ADX confirms trend strength — filter out range conditions where mean-reversion fails
3. Volume spike validates momentum shift — confirms institutional interest
4. 1d HTF trend keeps entries aligned with higher timeframe direction

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull market: KAMA rising = strong uptrend, ride the move
- Bear market: KAMA falling = strong downtrend, short the rallies
- Range/transition: KAMA flat + ADX low = no trade, skip chop

KEY INSIGHT: KAMA is the adaptive foundation used in SOLUSDT test Sharpe 1.31.
This is a SIMPLIFICATION of the KAMA strategy that works — less conditions = fewer trades.

TARGET: 80-150 total trades over 4 years (20-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=21, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average
    - ER (Efficiency Ratio) = direction / volatility
    - Fast SC = 2/(fast+1), Slow SC = 2/(slow+1)
    - KAMA = previous KAMA + ER * (price - previous KAMA) * SC
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio over 'period' bars
    direction = np.zeros(n)
    volatility = np.zeros(n)
    
    for i in range(period, n):
        d = close[i] - close[i - period]
        for j in range(i - period + 1, i + 1):
            volatility[i] += abs(close[j] - close[j - 1])
        
        direction[i] = abs(d)
        if volatility[i] > 0:
            er = direction[i] / volatility[i]
        else:
            er = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    smoothing = 2.0 / (period + 1)
    
    kama = np.zeros(n)
    kama[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        er = direction[i] / volatility[i] if volatility[i] > 0 else 0
        sc = er * (fast_sc - slow_sc) + slow_sc
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - trend strength filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DX
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            di_plus[i] = 100 * plus_dm_smooth[i] / atr[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr[i]
            
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def calculate_hma(data, period):
    """Hull Moving Average for smoother trend"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    # First HMA
    wma1 = np.zeros(n)
    for i in range(half - 1, n):
        window = half
        wma1[i] = np.sum(data[i - window + 1:i + 1] * np.arange(1, window + 1)) / (window * (window + 1) / 2)
    
    # Second HMA
    wma2 = np.zeros(n)
    for i in range(period - 1, n):
        window = period
        wma2[i] = np.sum(data[i - window + 1:i + 1] * np.arange(1, window + 1)) / (window * (window + 1) / 2)
    
    # Final HMA
    hma = np.zeros(n)
    for i in range(period - 1 + half, n):
        diff = wma1[i] - wma2[i]
        window = sqrt_n
        hma[i] = np.sum(data[i - window + 1:i + 1] * np.arange(1, window + 1)) / (window * (window + 1) / 2)
    
    # Simpler approach using pandas
    hma_series = pd.Series(data).rolling(window=period).apply(
        lambda x: pd.Series(x).rolling(window=half).mean().iloc[-1], raw=False
    )
    
    # Use EWM-based approximation
    return pd.Series(data).ewm(span=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === HTF Indicators (1d) ===
    # KAMA on 1d for trend direction
    kama_1d = calculate_kama(close_1d, period=21, fast=2, slow=30)
    
    # HTF: KAMA rising (bull) vs falling (bear)
    kama_1d_prev = np.roll(kama_1d, 1)
    kama_1d_prev[0] = kama_1d[0]
    
    # 1d KAMA trend: 1 = bull, -1 = bear, 0 = neutral
    htf_trend = np.zeros(len(kama_1d))
    for i in range(1, len(kama_1d)):
        if not np.isnan(kama_1d[i]) and not np.isnan(kama_1d_prev[i]):
            if kama_1d[i] > kama_1d_prev[i] * 1.001:  # 0.1% minimum change
                htf_trend[i] = 1  # Bull
            elif kama_1d[i] < kama_1d_prev[i] * 0.999:
                htf_trend[i] = -1  # Bear
    
    htf_trend_aligned = align_htf_to_ltf(prices, df_1d, htf_trend)
    
    # === Local 4h Indicators ===
    kama = calculate_kama(close, period=21, fast=2, slow=30)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # KAMA momentum (direction change)
    kama_prev = np.roll(kama, 1)
    kama_prev[0] = kama[0]
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size - balance between safety and opportunity
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # KAMA needs ~30, ADX needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === KAMA DIRECTION ===
        kama_rising = kama[i] > kama_prev[i] * 1.001
        kama_falling = kama[i] < kama_prev[i] * 0.999
        kama_flat = not kama_rising and not kama_falling
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx[i] > 22  # Must have minimum trend strength
        weak_trend = adx[i] < 18  # Weak = skip (range)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.6  # 60% above average
        
        # === HTF TREND ===
        htf_bull = htf_trend_aligned[i] > 0.5 if not np.isnan(htf_trend_aligned[i]) else False
        htf_bear = htf_trend_aligned[i] < -0.5 if not np.isnan(htf_trend_aligned[i]) else False
        htf_neutral = not htf_bull and not htf_bear
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY ===
            # Must have: KAMA rising + strong trend + volume + HTF bull or neutral
            if kama_rising and strong_trend and vol_spike:
                if htf_bull or htf_neutral:
                    desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Must have: KAMA falling + strong trend + volume + HTF bear or neutral
            elif kama_falling and strong_trend and vol_spike:
                if htf_bear or htf_neutral:
                    desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.5 ATR) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if KAMA flattens or falls
                if kama_flat or kama_falling:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if KAMA flattens or rises
                if kama_flat or kama_rising:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 6 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals