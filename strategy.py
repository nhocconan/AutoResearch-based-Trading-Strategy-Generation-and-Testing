#!/usr/bin/env python3
"""
Experiment #362: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Pullback

Hypothesis: Previous 4h strategy (#358) failed with Sharpe=-1.041 due to overly 
complex entry conditions (CRSI + Choppiness + Donchian = too many filters).

This version SIMPLIFIES dramatically:
1. KAMA (Kaufman Adaptive) instead of HMA - adapts to volatility, fewer whipsaws
2. RSI(14) pullback entries only - simpler than CRSI, more reliable
3. 1d HMA for directional bias ONLY (not entry trigger)
4. Loosened RSI thresholds (30/70 instead of 25/75) for MORE trades
5. Volume confirmation at 1.0x (not 1.2x) - easier to trigger
6. Simple ATR stoploss (2.5x)

Key difference from failed strategies:
- MAX 3 confluence factors per entry (not 5+)
- No Choppiness Index (unreliable on 4h)
- No Donchian breakouts (too many false signals)
- RSI pullback IN DIRECTION of HTF trend only

Regime Detection (SIMPLE):
- 1d HMA bull = only look for longs
- 1d HMA bear = only look for shorts
- No neutral regime (forces directional bias)

Entry Logic:
- Long: 1d HMA bull + KAMA(21) bull + RSI(14) crosses above 30 from below
- Short: 1d HMA bear + KAMA(21) bear + RSI(14) crosses below 70 from above

Position sizing: 0.25 base, 0.30 when volume confirms
Stoploss: 2.5x ATR(14) from entry price

Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
Timeframe: 4h (20-50 trades/year target per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_pullback_1d1w_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    # Efficiency Ratio (ER)
    er = np.zeros(n)
    er[:] = np.nan
    
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i-period:i+1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Smoothing constant
    sc = np.zeros(n)
    sc[:] = np.nan
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average for HTF trend"""
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

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, period=21, fast_period=2, slow_period=30)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    # RSI cross tracking
    prev_rsi = np.nan
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(kama_4h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        # === HTF BIAS (1d + 1w) ===
        # Strong bull: both 1d and 1w HMA bull
        # Bull: 1d HMA bull (1w neutral or bull)
        # Bear: 1d HMA bear (1w neutral or bear)
        # Strong bear: both 1d and 1w HMA bear
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # Determine directional bias
        htf_bull = htf_1d_bull
        htf_bear = htf_1d_bear
        
        # Strengthen bias if 1w agrees
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # === 4h KAMA TREND ===
        kama_bull = close[i] > kama_4h[i]
        kama_bear = close[i] < kama_4h[i]
        
        # === RSI PULLBACK DETECTION ===
        # Long: RSI was < 30, now crosses above 30
        # Short: RSI was > 70, now crosses below 70
        rsi_cross_long = False
        rsi_cross_short = False
        
        if not np.isnan(prev_rsi) and not np.isnan(rsi[i]):
            if prev_rsi < 30.0 and rsi[i] >= 30.0:
                rsi_cross_long = True
            if prev_rsi > 70.0 and rsi[i] <= 70.0:
                rsi_cross_short = True
        
        prev_rsi = rsi[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] >= vol_sma[i]  # >= 1.0x (loose)
        
        # === ENTRY LOGIC (SIMPLE - max 3 conditions) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + KAMA bull + RSI cross above 30
        if htf_bull and kama_bull and rsi_cross_long:
            desired_signal = SIZE_STRONG if (vol_confirm or htf_strong_bull) else SIZE_BASE
        
        # SHORT: HTF bear + KAMA bear + RSI cross below 70
        elif htf_bear and kama_bear and rsi_cross_short:
            desired_signal = -SIZE_STRONG if (vol_confirm or htf_strong_bear) else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals