#!/usr/bin/env python3
"""
Experiment #451: 6h Primary + 1d/1w HTF — Simplified Trend + Volume Confirmation

Hypothesis: 6h failures stem from OVER-COMPLEXITY (dual HTF, complex regime detection).
Previous 6h attempts (#443, #447) show:
- Fisher + Chop regime: Sharpe=-1.054 (too many false signals in chop)
- CRSI + weighted HTF: Sharpe=0.130 (barely positive, too few trades)
- Dual HTF (12h+1d): Too restrictive, misses trends when HTFs disagree temporarily

NEW APPROACH:
1. SINGLE HTF BIAS: 1d HMA for trend direction (not dual 12h+1d)
2. 1w MAJOR FILTER: Only use 1w for extreme bias (price vs 1w HMA > 5%)
3. VOLUME CONFIRMATION: Taker buy volume spike confirms breakout validity
4. SIMPLER ENTRY: 6h HMA cross + RSI momentum (not complex regime switching)
5. ATR TRAILING STOP: Dynamic stop based on volatility (2.5x ATR)
6. LOOSER THRESHOLDS: RSI 45/55 for momentum (not 40/60 extremes)

Why this should work on 6h:
- 6h captures multi-day swings without 4h noise
- 1d trend filter reduces whipsaw (proven on 4h strategies)
- Volume confirmation filters false breakouts (critical on HTF)
- Simpler logic = more trades qualify (target 40-60/year)

Entry Logic:
- Long: 1d HMA bull + 6h HMA cross up + RSI > 50 + volume spike
- Short: 1d HMA bear + 6h HMA cross down + RSI < 50 + volume spike
- 1w filter: Skip longs if price > 1w HMA + 8% (overextended), skip shorts if < -8%

Target: Sharpe>0.45, DD>-35%, trades>=60 train, trades>=10 test
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_simplified_trend_volume_1d1w_v1"
timeframe = "6h"
leverage = 1.0

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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def calculate_momentum(close, period=10):
    """Rate of Change / Momentum"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    momentum = np.zeros(n)
    momentum[:] = np.nan
    for i in range(period, n):
        if close[i-period] > 1e-10:
            momentum[i] = 100.0 * (close[i] - close[i-period]) / close[i-period]
    
    return momentum

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values if "taker_buy_volume" in prices.columns else volume * 0.5
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    hma_6h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    momentum = calculate_momentum(close, period=10)
    
    # Taker buy ratio (buying pressure)
    taker_ratio = np.zeros(n)
    taker_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            taker_ratio[i] = taker_buy_vol[i] / volume[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_sma[i]) or np.isnan(taker_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1D TREND BIAS (single HTF, not dual) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 1W MAJOR FILTER (overextension check) ===
        # Skip longs if price > 1w HMA + 8% (overbought on weekly)
        # Skip shorts if price < 1w HMA - 8% (oversold on weekly)
        hma_1w_val = hma_1w_aligned[i]
        price_vs_1w = (close[i] - hma_1w_val) / hma_1w_val * 100.0 if hma_1w_val > 1e-10 else 0.0
        
        overextended_long = price_vs_1w > 8.0
        overextended_short = price_vs_1w < -8.0
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === HMA CROSSOVER (fast vs slow) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_6h_fast[i]) and not np.isnan(hma_6h_fast[i-1]):
            if not np.isnan(hma_6h[i]) and not np.isnan(hma_6h[i-1]):
                if hma_6h_fast[i-1] <= hma_6h[i-1] and hma_6h_fast[i] > hma_6h[i]:
                    hma_cross_long = True
                if hma_6h_fast[i-1] >= hma_6h[i-1] and hma_6h_fast[i] < hma_6h[i]:
                    hma_cross_short = True
        
        # === VOLUME SPIKE CONFIRMATION ===
        # Volume must be > 1.3x 20-bar average for breakout validity
        volume_spike = volume[i] > 1.3 * vol_sma[i] if vol_sma[i] > 1e-10 else False
        
        # === TAKER BUY RATIO (buying/selling pressure) ===
        # > 0.55 = bullish pressure, < 0.45 = bearish pressure
        taker_bullish = taker_ratio[i] > 0.55
        taker_bearish = taker_ratio[i] < 0.45
        
        # === RSI MOMENTUM (LOOSENED: 45/55 instead of 40/60) ===
        rsi_bullish = rsi[i] > 50.0
        rsi_bearish = rsi[i] < 50.0
        rsi_strong_bull = rsi[i] > 55.0
        rsi_strong_bear = rsi[i] < 45.0
        
        # === MOMENTUM CONFIRMATION ===
        mom_bullish = momentum[i] > 0.0 if not np.isnan(momentum[i]) else False
        mom_bearish = momentum[i] < 0.0 if not np.isnan(momentum[i]) else False
        
        # === ENTRY LOGIC (SIMPLIFIED - 3 conditions max) ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bull + 6h HMA cross + volume spike + RSI/momentum confirm
        if htf_1d_bull and not overextended_long:
            # Need: HMA cross OR (HMA bull + volume + taker buy)
            if hma_cross_long:
                if volume_spike or taker_bullish:
                    desired_signal = SIZE_STRONG
            elif hma_bull and volume_spike and taker_bullish:
                if rsi_strong_bull or mom_bullish:
                    desired_signal = SIZE_BASE
        
        # SHORT ENTRY: 1d bear + 6h HMA cross + volume spike + RSI/momentum confirm
        elif htf_1d_bear and not overextended_short:
            # Need: HMA cross OR (HMA bear + volume + taker sell)
            if hma_cross_short:
                if volume_spike or taker_bearish:
                    desired_signal = -SIZE_STRONG
            elif hma_bear and volume_spike and taker_bearish:
                if rsi_strong_bear or mom_bearish:
                    desired_signal = -SIZE_BASE
        
        # === TRAILING STOPLOSS CHECK (2.5x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest price for trailing
            if close[i] > highest_price:
                highest_price = close[i]
            # Trailing stop: highest - 2.5*ATR
            trailing_stop = highest_price - 2.5 * atr[i]
            if low[i] < trailing_stop or low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Update lowest price for trailing
            if close[i] < lowest_price:
                lowest_price = close[i]
            # Trailing stop: lowest + 2.5*ATR
            trailing_stop = lowest_price + 2.5 * atr[i]
            if high[i] > trailing_stop or high[i] > stop_price:
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
                # Set initial stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                    highest_price = entry_price
                else:
                    stop_price = entry_price + 2.5 * entry_atr
                    lowest_price = entry_price
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = final_signal
    
    return signals