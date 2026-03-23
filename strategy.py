#!/usr/bin/env python3
"""
Experiment #1216: 12h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 12h strategies failed due to overly complex regime filters
(Choppiness + Donchian + KAMA + multiple HTF). Simpler HMA crossover + RSI pullback
has shown promise in literature and should generate 30-50 trades/year on 12h.

Key design:
- HMA(16)/HMA(48) crossover on 12h for clean trend signal (less lag than EMA)
- RSI(14) pullback entries: long when RSI 35-50 in uptrend, short when 50-65 in downtrend
- ADX(14) > 18 filter to avoid choppy whipsaws (looser than 25 for more trades)
- 1d HMA(21) for macro trend bias (only trade with higher TF direction)
- ATR(14) 2.5x trailing stop for risk management
- Position size: 0.30 discrete (conservative for 12h)

Why this should work:
- HMA crossover is proven trend follower with less lag
- RSI pullback entries catch retracements in trends (not extremes)
- ADX > 18 is loose enough to generate trades but filters dead chop
- 1d HMA bias prevents counter-trend trades in strong macro moves
- Simpler logic = fewer conflicting filters = more consistent signals

Target: Sharpe > 0.612, trades >= 30 on train, >= 3 on test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crossover_rsi_pullback_1d_adx_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        """Weighted Moving Average."""
        weights = np.arange(1, span + 1, dtype=float)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # Calculate 2*WMA(half) - WMA(full)
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    # Apply WMA with sqrt(period) to the difference
    for i in range(period - 1 + sqrt_period - 1, n):
        window = diff[i - sqrt_period + 1:i + 1]
        if not np.any(np.isnan(window)) and len(window) == sqrt_period:
            weights = np.arange(1, sqrt_period + 1, dtype=float)
            hma[i] = np.sum(window * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0.0)
    loss[1:] = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = choppy/range
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with alpha=1/period)
    atr = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    # Initialize with SMA
    atr[period-1] = np.mean(tr[:period])
    plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
    
    # Wilder's smoothing
    for i in range(period, n):
        atr[i] = atr[i-1] * (1 - 1/period) + tr[i] * (1/period)
        plus_dm_smooth[i] = plus_dm_smooth[i-1] * (1 - 1/period) + plus_dm[i] * (1/period)
        minus_dm_smooth[i] = minus_dm_smooth[i-1] * (1 - 1/period) + minus_dm[i] * (1/period)
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    mask = atr > 1e-10
    di_plus[mask] = 100.0 * plus_dm_smooth[mask] / atr[mask]
    di_minus[mask] = 100.0 * minus_dm_smooth[mask] / atr[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    for i in range(period * 2 - 1, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period * 2 - 1] = np.mean(dx[period-1:period*2])
    for i in range(period * 2, n):
        adx[i] = adx[i-1] * (1 - 1/period) + dx[i] * (1/period)
    
    return adx

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
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (HMA Crossover) ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # === TREND STRENGTH (ADX) ===
        trending = adx[i] > 18.0  # Loose threshold for more trades
        
        # === RSI PULLBACK ZONES ===
        # In uptrend: look for RSI pullback to 35-50 zone
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        # In downtrend: look for RSI pullback to 45-65 zone
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG: HMA bull + macro not bearish + trending + RSI pullback
        if hma_bull and not macro_bear and trending and rsi_pullback_long:
            desired_signal = BASE_SIZE
        
        # SHORT: HMA bear + macro not bullish + trending + RSI pullback
        elif hma_bear and not macro_bull and trending and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
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
                # Position flip
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