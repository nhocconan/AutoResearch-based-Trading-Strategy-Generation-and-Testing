#!/usr/bin/env python3
"""
Experiment #671: 4h Primary + 1d HTF — KAMA Trend + ADX Regime + RSI Entries

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to volatility better than 
HMA/EMA, reducing whipsaws in choppy markets. Combined with ADX regime detection 
(hysteresis: enter >25, exit <18) and loose RSI thresholds (30/70) for entries, 
this should generate 30-50 trades/year with positive Sharpe across ALL symbols.

Key innovations vs failed strategies:
1. KAMA instead of HMA — ER (Efficiency Ratio) adapts to trending vs noisy
2. ADX hysteresis prevents regime flip-flopping
3. LOOSE RSI thresholds (35/65) ensure trade generation (learned from 0-trade failures)
4. 1d HTF for macro bias — prevents counter-trend entries
5. ATR trailing stop (2.5x) for risk management
6. Discrete position sizing (0.25/0.30) to minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency (trending vs noisy)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = np.abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi[period:] = 100 - (100 / (1 + rs[period-1:]))
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index with +DI/-DI"""
    n = len(close)
    adx = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    if n < period * 2:
        return adx, plus_di, minus_di
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    smooth_plus_dm = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    smooth_minus_dm = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI calculations
    plus_di[period:] = 100 * smooth_plus_dm[period:] / (atr[period:] + 1e-10)
    minus_di[period:] = 100 * smooth_minus_dm[period:] / (atr[period:] + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx[period*2:] = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values[period*2:]
    
    return adx, plus_di, minus_di

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators
    kama_4h = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    rsi_4h = calculate_rsi(close, period=14)
    adx_4h, plus_di_4h, minus_di_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    # Calculate 1d KAMA and align to 4h
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX hysteresis state
    prev_adx_regime = 0  # 0=neutral, 1=trend, -1=chop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(rsi_4h[i]):
            continue
        if np.isnan(adx_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === ADX REGIME WITH HYSTERESIS ===
        adx_value = adx_4h[i]
        
        if prev_adx_regime <= 0 and adx_value > 25:
            adx_regime = 1  # Enter trend regime
        elif prev_adx_regime >= 0 and adx_value < 18:
            adx_regime = -1  # Enter chop regime
        else:
            adx_regime = prev_adx_regime  # Maintain current regime
        
        prev_adx_regime = adx_regime
        
        # === HTF TREND BIAS (1d KAMA) ===
        htf_bullish = close[i] > kama_1d_aligned[i]
        htf_bearish = close[i] < kama_1d_aligned[i]
        
        # === 4h TREND (KAMA) ===
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # === RSI SIGNALS (LOOSE thresholds) ===
        rsi_oversold = rsi_4h[i] < 35
        rsi_overbought = rsi_4h[i] > 65
        rsi_neutral = 35 <= rsi_4h[i] <= 65
        
        # === DI DIRECTION ===
        di_bullish = plus_di_4h[i] > minus_di_4h[i] if not np.isnan(plus_di_4h[i]) else False
        di_bearish = minus_di_4h[i] > plus_di_4h[i] if not np.isnan(minus_di_4h[i]) else False
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (ADX > 25 or maintained) ===
        if adx_regime == 1:
            # Long: HTF bullish + 4h KAMA bullish + DI bullish + RSI not overbought
            if htf_bullish and kama_bullish and di_bullish and rsi_4h[i] < 70:
                desired_signal = SIZE_LONG
            # Short: HTF bearish + 4h KAMA bearish + DI bearish + RSI not oversold
            elif htf_bearish and kama_bearish and di_bearish and rsi_4h[i] > 30:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: CHOPPY (ADX < 18 or maintained) ===
        elif adx_regime == -1:
            # Mean reversion: RSI extremes with HTF filter
            if rsi_oversold and not htf_bearish:
                desired_signal = SIZE_LONG
            elif rsi_overbought and not htf_bullish:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION ===
        else:
            # Use KAMA direction with RSI filter
            if kama_bullish and rsi_4h[i] < 60:
                desired_signal = SIZE_LONG
            elif kama_bearish and rsi_4h[i] > 40:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                if kama_bullish and rsi_4h[i] < 75:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                if kama_bearish and rsi_4h[i] > 25:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
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
    
    return signals