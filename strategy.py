#!/usr/bin/env python3
"""
Experiment #1599: 4h Primary + 1d HTF — Choppiness Regime + Dual Strategy

Hypothesis: Single-regime strategies fail because markets alternate between trending and chopping.
By detecting regime with Choppiness Index and switching logic accordingly, we capture both:
- Trending (CHOP < 38.2): Donchian breakout + HMA trend confirmation
- Choppy (CHOP > 61.8): RSI mean reversion at extremes

Key innovations:
1. Choppiness Index(14) regime detection on 4h
2. 1d HMA(21) for higher-timeframe trend bias (call ONCE before loop)
3. Dual entry logic based on regime (trend vs mean-revert)
4. RSI(14) extremes for mean-reversion entries (<30 long, >70 short)
5. Donchian(20) breakouts for trend entries
6. ATR(14) 2.5x trailing stop for drawdown control
7. Position size 0.25-0.30 discrete levels

Why this should beat Sharpe 0.618:
- Regime switching captures more market states than single-strategy approaches
- Looser RSI filters (30/70 vs 45-55) ensure we generate trades
- 1d HMA bias prevents counter-trend trades in strong trends
- 4h targets 20-50 trades/year with regime switching
- Proven pattern: ETH Sharpe +0.923 with Choppiness + Connors RSI

Timeframe: 4h (required)
HTF: 1d HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_donchian_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP = 100 * (SUM(ATR, n) / (Highest High - Lowest Low)) * (1 / log10(n))
    
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    choppiness = np.full(n, np.nan)
    mask = (hh - ll) > 1e-10
    choppiness[mask] = 100.0 * (atr_sum[mask] / (hh[mask] - ll[mask])) * (1.0 / np.log10(period))
    
    return choppiness

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # HMA for trend confirmation
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
        if np.isnan(choppiness[i]) or np.isnan(rsi[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = choppiness[i] > 55.0  # Midpoint for regime switch
        is_trending = choppiness[i] < 45.0
        
        # === TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND (HMA slope) ===
        hma_bull = hma_4h[i] > hma_4h[i-10] if not np.isnan(hma_4h[i-10]) else daily_bull
        hma_bear = hma_4h[i] < hma_4h[i-10] if not np.isnan(hma_4h[i-10]) else daily_bear
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND FOLLOWING MODE: Donchian breakout + trend confirmation
            # Long: Price breaks Donchian upper + daily bull + 4h HMA bull
            if close[i] >= donchian_upper[i] and daily_bull and hma_bull:
                desired_signal = BASE_SIZE
            
            # Short: Price breaks Donchian lower + daily bear + 4h HMA bear
            elif close[i] <= donchian_lower[i] and daily_bear and hma_bear:
                desired_signal = -BASE_SIZE
        
        elif is_choppy:
            # MEAN REVERSION MODE: RSI extremes + counter to daily trend for shorts
            # Long: RSI < 30 (oversold) + daily not strongly bear
            if rsi[i] < 32.0 and not (daily_bear and close[i] < hma_4h[i]):
                desired_signal = BASE_SIZE
            
            # Short: RSI > 68 (overbought) + daily not strongly bull
            elif rsi[i] > 68.0 and not (daily_bull and close[i] > hma_4h[i]):
                desired_signal = -BASE_SIZE
        
        else:
            # TRANSITION REGIME: Use RSI with trend bias
            # Long: RSI < 40 + daily bull
            if rsi[i] < 40.0 and daily_bull:
                desired_signal = BASE_SIZE * 0.5
            
            # Short: RSI > 60 + daily bear
            elif rsi[i] > 60.0 and daily_bear:
                desired_signal = -BASE_SIZE * 0.5
        
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
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.5
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